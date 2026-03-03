# import os, glob
from tqdm import tqdm
# from gigapath.pipeline import tile_one_slide

# # Please set your Hugging Face API token
# os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

# assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"


# if __name__ == '__main__':

#     SLIDE_ROOT='/pscratch/sd/j/jehr/MSFT/CoreBT/data/corebt_pathology'
#     local_dir='/pscratch/sd/j/jehr/MSFT/CoreBT/corebt-baseline/prov-gigapath'
#     level=0

#     slides = sorted(glob.glob(os.path.join(SLIDE_ROOT, '*.svs')))

#     for ix, slide_path in tqdm(enumerate(slides), total = len(slides), colour='red'):
#         print(f'\n\nTiling slide {slide_path}')
#         subject_id=slide_path.split('/')[-1].split('.svs')[0]
#         save_dir = os.path.join(local_dir, f'outputs/preprocessing-corebt/{subject_id}_lv{level}/')
#         tile_one_slide(slide_path, save_dir=save_dir, level=level)
#         print(f'** Tiled slide, saved to {save_dir}')

#!/usr/bin/env python3

import sys
import os
import traceback
import logging

def main():
    if len(sys.argv) < 4:
        print(
            "Usage: tile_single_slide.py <slide_path> <save_dir> <level>",
            file=sys.stderr,
        )
        sys.exit(2)

    slide_path = sys.argv[1]
    save_dir = sys.argv[2]
    level = int(sys.argv[3])

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    logging.info(f"Tiling slide: {slide_path}")
    logging.info(f"Output dir: {save_dir}")
    logging.info(f"Level: {level}")

    try:
        # Import here so crashes don't poison parent
        from gigapath.pipeline import tile_one_slide

        os.makedirs(save_dir, exist_ok=True)

        tile_one_slide(
            slide_path,
            save_dir=save_dir,
            level=level,
        )

        logging.info("Slide completed successfully")
        sys.exit(0)

    except RuntimeError as e:
        # Catch CUDA / torch OOMs
        if "out of memory" in str(e).lower():
            logging.error("OOM during slide tiling")
        else:
            logging.error("RuntimeError during slide tiling")
            traceback.print_exc()
        sys.exit(1)

    except MemoryError:
        logging.error("Python MemoryError during slide tiling")
        sys.exit(1)

    except Exception:
        logging.error("Unexpected exception during slide tiling")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
