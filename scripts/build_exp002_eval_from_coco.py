import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Build EXP-002 eval jsonl from COCO captions')
    parser.add_argument('--captions-json', type=str, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=20)
    args = parser.parse_args()

    captions_path = Path(args.captions_json)
    images_dir = Path(args.images_dir).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with captions_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    # pick first caption per image for stable baseline
    picked = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id in picked:
            continue
        filename = image_id_to_filename.get(img_id)
        if not filename:
            continue
        image_path = images_dir / filename
        if not image_path.exists():
            continue
        caption = ann.get('caption', '').strip()
        if not caption:
            continue
        picked[img_id] = {
            'sample_id': f'coco-{img_id}',
            'image': str(image_path.resolve()),
            'prompt': 'Describe this image in one short sentence.',
            'reference': caption,
            'subsets': ['scene_caption_proxy'],
        }
        if len(picked) >= args.max_samples:
            break

    with output.open('w', encoding='utf-8') as f:
        for _, item in picked.items():
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Wrote {len(picked)} samples to {output}')


if __name__ == '__main__':
    main()
