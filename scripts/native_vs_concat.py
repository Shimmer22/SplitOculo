#!/usr/bin/env python3
"""Qwen Native Video API vs Qwen Per-Frame Image Concat (no compression).
Native: HF video API (temporal patch merging, joint ViT, cross-frame attention)
Concat: per-frame image ViT → concat vision embeddings → LLM (no cross-frame ViT attn)"""
import subprocess, sys, time, threading, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
for _a,_t in [('long',np.int64),('longlong',np.int64),('ulong',np.uint64),('ulonglong',np.uint64),('uintc',np.uint32)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import torch
_t=torch.tensor; _dm={np.uint8:torch.uint8,np.int32:torch.int32,np.int64:torch.int64,np.float32:torch.float32}
torch.from_numpy=lambda x:_t(x.tolist(),dtype=_dm.get(x.dtype.type,torch.float32))

from PIL import Image
from core.qwen_extractor import QwenFeatureExtractor
from transformers import TextIteratorStreamer

VIDEO="/data/downloads/20260115_0001_dajiangaction5pro_lingshouxiaofei_chaoshi_0001.MP4"
PROMPT="These are uniformly sampled first-person supermarket video frames. Question: During the middle part of the video, the wearer selected or picked items in front of which product area? Answer in one short Chinese noun phrase. Describe only visible product category or shelf area."
MAX_TOKENS=32

def read_uniform(vid,nf):
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','default=noprint_wrappers=1',vid],capture_output=True,text=True)
    info={}; [info.update({k:v}) for line in probe.stdout.strip().split('\n') if '=' in line for k,v in [line.split('=',1)]]
    total=int(info.get('nb_frames',0)); w,h=int(info.get('width',1920)),int(info.get('height',1080))
    idx=np.linspace(0,total-1,nf,dtype=int); sel='+'.join(f'eq(n,{i})' for i in idx)
    proc=subprocess.Popen(['ffmpeg','-v','error','-i',vid,'-vf',f"select='{sel}'",'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    fs=w*h*3; fr=[]
    while True:
        raw=proc.stdout.read(fs)
        if len(raw)<fs: break
        fr.append(Image.frombytes('RGB',(w,h),raw))
    proc.wait(); return fr

def b2d(b,d): return {k:v.to(d) if hasattr(v,"to") else v for k,v in b.items()}

@torch.no_grad()
def native_video(ext, frames):
    msgs=[{"role":"user","content":[{"type":"video","video":"sampled_frames"},{"type":"text","text":PROMPT}]}]
    txt=ext.processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=ext.processor(text=[txt],videos=[frames],return_tensors="pt",padding=True); inp=b2d(inp,ext.device)
    t0=time.perf_counter()
    s=TextIteratorStreamer(ext.processor.tokenizer,skip_prompt=True,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    out={}
    t=threading.Thread(target=lambda: out.update({"o":ext.model.generate(**inp,max_new_tokens=MAX_TOKENS,do_sample=False,streamer=s)}))
    t.start(); ch=[]; ft=None
    for c in s:
        if c and ft is None: ft=time.perf_counter()-t0
        ch.append(c)
    t.join(); total=time.perf_counter()-t0
    return "".join(ch).strip(),ft,total

@torch.no_grad()
def frame_concat(ext, frames):
    """Per-frame Qwen ViT (full 32 layers + merger) → concat pooler_output → LLM."""
    vis_embs=[]; vi_time=0.0
    for frame in frames:
        msgs=[{"role":"user","content":[{"type":"image","image":frame},{"type":"text","text":"."}]}]
        txt=ext.processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=ext.processor(text=[txt],images=[frame],return_tensors="pt",padding=True); inp=b2d(inp,ext.device)
        t_vi=time.perf_counter()
        vis=ext.model.model.get_image_features(pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
        vi_time+=time.perf_counter()-t_vi
        vis_embs.append(vis.pooler_output[0].cpu())

    visual_tokens=torch.stack(vis_embs,dim=0)
    print(f"  Concat: {visual_tokens.shape}, ViT: {vi_time:.3f}s")

    # Inject into LLM using video pad tokens
    model=ext.model; vis_id=model.config.video_token_id
    num_vis=visual_tokens.shape[0]*visual_tokens.shape[1]
    ph="<|vision_start|>"+"<|video_pad|>"*num_vis+"<|vision_end|>"
    msgs=[{'role':'user','content':ph+PROMPT}]
    txt=ext.processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    tin=ext.processor.tokenizer(txt,return_tensors='pt',padding=True)
    ids=tin['input_ids'].to(ext.device); am=tin['attention_mask'].to(ext.device)
    emb=model.get_input_embeddings(); ie=emb(ids)
    vm=(ids==vis_id)
    vf=visual_tokens.view(-1,visual_tokens.shape[-1])
    nv=vm.sum().item()
    if vf.shape[0]<nv: vf=torch.cat([vf,vf[-1:].repeat(nv-vf.shape[0],1)],dim=0)
    elif vf.shape[0]>nv: vf=vf[:nv]
    vf=vf.to(ie.dtype)
    bi,ti=torch.where(vm)
    for i,(b,tt) in enumerate(zip(bi,ti)): ie[b,tt]=vf[i]

    t0=time.perf_counter()
    s=TextIteratorStreamer(ext.processor.tokenizer,skip_prompt=True,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    out={}
    t=threading.Thread(target=lambda: out.update({"o":model.generate(inputs_embeds=ie,attention_mask=am,max_new_tokens=MAX_TOKENS,do_sample=False,streamer=s,pad_token_id=ext.processor.tokenizer.pad_token_id,eos_token_id=ext.processor.tokenizer.eos_token_id)}))
    t.start(); ch=[]; ft=None
    for c in s:
        if c and ft is None: ft=time.perf_counter()-t0
        ch.append(c)
    t.join(); total=time.perf_counter()-t0
    ans="".join(ch).strip()
    if 'assistant' in ans.lower(): ans=ans.split('assistant')[-1].strip()
    return ans, ft, total+vi_time  # total includes ViT + LLM


if __name__=="__main__":
    print("Loading Qwen...")
    ext=QwenFeatureExtractor(min_pixels=224*224,max_pixels=448*448,device="cuda",local_files_only=True,extract_layer=4).load()

    results=[]
    for nf in [2,4,8,16,32,64]:
        print(f"\n{'='*50}\n{nf} frames")
        fr=read_uniform(VIDEO,nf)
        print(f"Read {len(fr)} frames")

        a1,ft1,t1=native_video(ext,fr)
        print(f"  NATIVE:  {a1:<14} FT={ft1:.3f}s T={t1:.3f}s")

        try:
            a2,ft2,t2=frame_concat(ext,fr)
            print(f"  CONCAT:  {a2:<14} FT={ft2:.3f}s T={t2:.3f}s")
        except Exception as e:
            import traceback; print(f"  CONCAT ERROR: {traceback.format_exc()[:300]}")
            a2,ft2,t2=f"ERR",None,None

        same="✅" if a1==a2 else "❌"
        results.append({"nf":nf,"native":a1,"concat":a2,"nat_ft":ft1,"nat_tot":t1,"con_ft":ft2,"con_tot":t2,"same":same})

    print(f"\n{'='*50}\nSUMMARY")
    print(f"{'Frames':<8}{'Native':<14}{'Concat':<14}{'Same?':<8}{'Nat T':<10}{'Con T':<10}")
    for r in results:
        print(f"{r['nf']:<8}{r['native']:<14}{r['concat']:<14}{r['same']:<8}{r['nat_tot']:<10.3f}{r['con_tot']:<10.3f}")

    with open("/workspace/SplitOculo/native_vs_concat.json","w") as f: json.dump(results,f,indent=2,ensure_ascii=False)
