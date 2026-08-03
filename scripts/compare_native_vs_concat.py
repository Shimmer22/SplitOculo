#!/usr/bin/env python3
"""Compare Qwen Native Video vs Qwen Frame Concat (no compression).
Both use Qwen ViT — difference is temporal joint-processing vs per-frame independence."""
import subprocess, sys, time, threading, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
for _a,_t in [('long',np.int64),('longlong',np.int64),('ulong',np.uint64),('ulonglong',np.uint64),('uintc',np.uint32)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import torch
_t=torch.tensor; _dm={np.uint8:torch.uint8,np.int8:torch.int8,np.int16:torch.int16,np.int32:torch.int32,
    np.int64:torch.int64,np.float16:torch.float16,np.float32:torch.float32,np.float64:torch.float64,np.bool_:torch.bool}
torch.from_numpy=lambda x:_t(x.tolist(),dtype=_dm.get(x.dtype.type,torch.float32))

from PIL import Image
from core.qwen_extractor import QwenFeatureExtractor
from transformers import TextIteratorStreamer

VIDEO = "/data/downloads/20260115_0001_dajiangaction5pro_lingshouxiaofei_chaoshi_0001.MP4"
PROMPT = "These are uniformly sampled first-person supermarket video frames. Question: During the middle part of the video, the wearer selected or picked items in front of which product area? Answer in one short Chinese noun phrase. Describe only visible product category or shelf area."
MAX_NEW_TOKENS = 32
FRAME_COUNTS = [2, 4, 8, 16, 32, 64]


def read_uniform(video, nf):
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','default=noprint_wrappers=1',video],
        capture_output=True,text=True)
    info={}
    for line in probe.stdout.strip().split('\n'):
        if '=' in line: k,v=line.split('=',1); info[k]=v
    total=int(info.get('nb_frames',0)); w,h=int(info.get('width',1920)),int(info.get('height',1080))
    indices=np.linspace(0,total-1,nf,dtype=int).tolist()
    sel='+'.join(f'eq(n,{i})' for i in indices)
    proc=subprocess.Popen(['ffmpeg','-v','error','-i',video,'-vf',f"select='{sel}'",'-vsync','0',
        '-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    fs=w*h*3; frames=[]
    while True:
        raw=proc.stdout.read(fs)
        if len(raw)<fs: break
        frames.append(Image.frombytes('RGB',(w,h),raw))
    proc.wait()
    return frames

def batch_to_device(batch, dev):
    return {k:v.to(dev) if hasattr(v,"to") else v for k,v in batch.items()}

@torch.no_grad()
def qwen_native_infer(ext, frames):
    """Qwen built-in video API: all frames through ViT together."""
    messages=[{"role":"user","content":[{"type":"video","video":"sampled_frames"},{"type":"text","text":PROMPT}]}]
    text=ext.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=ext.processor(text=[text],videos=[frames],return_tensors="pt",padding=True)
    inputs=batch_to_device(inputs,ext.device)
    t0=time.perf_counter()
    streamer=TextIteratorStreamer(ext.processor.tokenizer,skip_prompt=True,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    out={}
    def _run(): out["o"]=ext.model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,streamer=streamer)
    t=threading.Thread(target=_run); t.start()
    chunks=[]; ft=None
    for chunk in streamer:
        if chunk and ft is None: ft=time.perf_counter()-t0
        chunks.append(chunk)
    t.join(); total=time.perf_counter()-t0
    ans="".join(chunks).strip()
    return ans,{"ft":ft,"total":total,"tokens":out["o"].shape[1]-inputs["input_ids"].shape[1]}

@torch.no_grad()
def qwen_concat_infer(ext, frames):
    """Per-frame feature extraction + concatenation into video grid.
    Each frame independently through ViT → concat → LLM (no cross-frame ViT attention)."""
    visual = ext.model.model.visual

    # Process per-frame: extract features at split layer 4
    per_frame_features = []
    per_frame_grid_hw = []
    for frame in frames:
        # Get Qwen-processed pixel values for this single frame
        messages=[{"role":"user","content":[{"type":"image","image":frame},{"type":"text","text":PROMPT}]}]
        text=ext.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=ext.processor(text=[text],images=[frame],return_tensors="pt",padding=True)
        pv=inputs["pixel_values"].to(ext.device)
        grid_thw=inputs.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw=grid_thw.to(ext.device)
            hw_dim = int(grid_thw[0,1].item()), int(grid_thw[0,2].item())
        else:
            hw_dim = int(pv.shape[0]**0.5), int(pv.shape[0]**0.5)

        # Run ViT up to layer 4
        hs = visual.patch_embed(pv.to(visual.patch_embed.proj.weight.dtype))
        rot = visual.rot_pos_emb(torch.tensor([[1,hw_dim[0],hw_dim[1]]],device=pv.device))
        seq_len = hs.shape[0]
        hs = hs.reshape(seq_len//visual.spatial_merge_unit, visual.spatial_merge_unit, -1)

        w_idx = torch.arange(seq_len//visual.spatial_merge_unit, device=pv.device)
        hs = hs[w_idx,:,:].reshape(seq_len,-1)

        rot = rot.reshape(seq_len//visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        rot = rot[w_idx,:,:].reshape(seq_len,-1)
        emb = torch.cat((rot,rot),dim=-1); pe=(emb.cos(),emb.sin())

        for layer_num in range(4):
            hs = visual.blocks[layer_num](hs, cu_seqlens=None, position_embeddings=pe)

        # Merge back
        seg = seq_len//visual.spatial_merge_unit
        hs = hs.view(seg, visual.spatial_merge_unit, -1)
        hs = hs[torch.argsort(w_idx),:,:].view(seq_len,-1)

        per_frame_features.append(hs.cpu())
        per_frame_grid_hw.append(hw_dim)

    # After layer 4, concat all frames and run layers 5-31 together
    all_features = torch.cat(per_frame_features, dim=0).to(ext.device)
    T = len(frames)
    H, W = per_frame_grid_hw[0]
    video_grid_thw = torch.tensor([[T,H,W]], device=ext.device)

    # Continue ViT from layer 4 onward
    rot = visual.rot_pos_emb(video_grid_thw)
    w_idx, cu_w = visual.get_window_index(video_grid_thw)
    cu_w = torch.tensor(cu_w, device=ext.device, dtype=torch.int32)
    cu_w = torch.unique_consecutive(cu_w)

    seq_len = all_features.shape[0]
    seg = seq_len // visual.spatial_merge_unit
    hs = all_features.reshape(seg, visual.spatial_merge_unit, -1)
    hs = hs[w_idx,:,:].reshape(seq_len, -1)

    rot = rot.reshape(seg, visual.spatial_merge_unit, -1)
    rot = rot[w_idx,:,:].reshape(seq_len, -1)
    emb = torch.cat((rot,rot),dim=-1); pe=(emb.cos(),emb.sin())

    cu_seqlens = torch.repeat_interleave(video_grid_thw[:,1]*video_grid_thw[:,2], video_grid_thw[:,0])
    cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens,(1,0),value=0)

    for layer_num, blk in enumerate(visual.blocks[4:], start=4):
        csl = cu_seqlens if layer_num in visual.fullatt_block_indexes else cu_w
        hs = blk(hs, cu_seqlens=csl, position_embeddings=pe)

    # Reverse window indexing
    rev = torch.argsort(w_idx)
    hs = hs.view(seg, visual.spatial_merge_unit, -1)
    hs = hs[rev,:,:].view(seq_len, -1)

    # Merger
    merger_out = visual.merger(hs)  # (total_tokens, 2048)

    # Feed to LLM via inputs_embeds
    t0 = time.perf_counter()
    messages=[{"role":"user","content":[{"type":"video","video":"sampled_frames"},{"type":"text","text":PROMPT}]}]
    text=ext.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    dummy_inputs=ext.processor(text=[text],videos=[frames],return_tensors="pt",padding=True)
    input_ids=dummy_inputs["input_ids"].to(ext.device)

    embed_layer=ext.model.model.embed_tokens if hasattr(ext.model.model,'embed_tokens') else ext.model.get_input_embeddings()
    text_embeds=embed_layer(input_ids)

    # Replace video tokens
    vis_token_id = ext.model.config.vision_token_id if hasattr(ext.model.config,'vision_token_id') else ext.model.config.image_token_id
    vis_mask = (input_ids[0]==vis_token_id)
    vis_indices = vis_mask.nonzero(as_tuple=False)
    if len(vis_indices)>0:
        v_start=vis_indices[0,0].item(); v_end=vis_indices[-1,0].item()+1
        if merger_out.shape[0]==(v_end-v_start):
            text_embeds[0,v_start:v_end]=merger_out.to(text_embeds.dtype)

    streamer=TextIteratorStreamer(ext.processor.tokenizer,skip_prompt=True,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    out={}
    def _run(): out["o"]=ext.model.generate(inputs_embeds=text_embeds,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,streamer=streamer)
    t=threading.Thread(target=_run); t.start()
    chunks=[]; ft=None
    for chunk in streamer:
        if chunk and ft is None: ft=time.perf_counter()-t0
        chunks.append(chunk)
    t.join(); total=time.perf_counter()-t0
    ans="".join(chunks).strip()
    return ans,{"ft":ft,"total":total,"tokens":out["o"].shape[1]-input_ids.shape[1]}


if __name__=="__main__":
    print("Pre-loading Qwen...")
    qext = QwenFeatureExtractor(min_pixels=224*224, max_pixels=448*448, device="cuda", local_files_only=True, extract_layer=4).load()

    results=[]
    for nf in FRAME_COUNTS:
        print(f"\n{'='*60}\nFRAMES: {nf}\n{'='*60}")
        frames = read_uniform(VIDEO, nf)
        print(f"Decoded {len(frames)} frames")

        # Native
        ans_n, tim_n = qwen_native_infer(qext, frames)
        print(f"\n  [NATIVE] {ans_n} | FT={tim_n['ft']:.3f}s Total={tim_n['total']:.3f}s Tokens={tim_n['tokens']}")

        # Concat
        try:
            ans_c, tim_c = qwen_concat_infer(qext, frames)
            print(f"  [CONCAT] {ans_c} | FT={tim_c['ft']:.3f}s Total={tim_c['total']:.3f}s Tokens={tim_c['tokens']}")
        except Exception as e:
            ans_c, tim_c = f"ERROR: {e}", {}
            print(f"  [CONCAT] ERROR: {e}")

        results.append({"frames":nf,"native":ans_n,"concat":ans_c,"native_ft":tim_n.get("ft"),"native_total":tim_n.get("total"),"concat_ft":tim_c.get("ft"),"concat_total":tim_c.get("total")})

    # Print table
    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Frames':<8}{'Native':<14}{'Concat':<14}{'Same?':<8}")
    print("-"*44)
    for r in results:
        same = "✅" if r["native"]==r["concat"] else "❌"
        print(f"{r['frames']:<8}{r['native']:<14}{r['concat']:<14}{same:<8}")

    # Save
    with open("/workspace/SplitOculo/native_vs_concat.json","w") as f:
        json.dump(results,f,indent=2,ensure_ascii=False)
    print(f"\nSaved: native_vs_concat.json")
