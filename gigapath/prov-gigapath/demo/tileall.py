import subprocess
import sys
import os, glob, argparse

'''
python3 -m demo.tileall >> tile5.log 2>&1
'''

SLIDE_ROOT='/gscratch/scrubbed/juampablo/corebt/corebt_pathology'
slide_save_dir='/gscratch/scrubbed/juampablo/corebt/corebt_pathology_tiles'
# gigapath_repo_root='/gscratch/kurtlab/models/gigapath/prov-gigapath'
level=0


def main(args):        
    # logging 
    os.makedirs(args.logdir, exist_ok=True)
    success_log = os.path.join(args.logdir, f"split_{args.split_no}_success.txt")
    failure_log = os.path.join(args.logdir, f"split_{args.split_no}_failed.txt")


    slides = sorted(glob.glob(os.path.join(SLIDE_ROOT, '*.svs')))


    # split into num_splits and only process split_no
    total = len(slides)
    chunk_size = (total + args.num_splits - 1) // args.num_splits 
    start = args.split_no * chunk_size
    end = min(start + chunk_size, total)
    slides = slides[start:end]

    print(
        f"Total number of slides: {total} | "
        f"Split {args.split_no + 1}/{args.num_splits} -> "
        f"{len(slides)} subjects"
    )

    for ix, slide_path in enumerate(slides):

        print(f'\n\n [{ix+1}/{len(slides)}] Tiling slide {slide_path}')

        subject_id = os.path.basename(slide_path).replace(".svs", "")
        save_dir = os.path.join(
            slide_save_dir,
            f"{subject_id}_lv{level}",
            # f"outputs/preprocessing-corebt/{subject_id}_lv{level}",
        )

        if os.path.exists(success_log): 
            with open(success_log, 'r') as f:
                if slide_path in f.read():
                    print(f"Skipping {slide_path}, already done.") # skip completed subjects. tile_single_slide already does this, but it is faster if we skip here.
                    continue

        cmd = [
            sys.executable,
            "demo/tile_single_slide.py",
            slide_path,
            save_dir,
            str(level),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()  # project root

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"[FAILED] {slide_path}, continuing")
            with open(failure_log, "a") as f:
                f.write(f"{slide_path}\n")
        else:
            print(f"[OK] {slide_path}")
            with open(success_log, "a") as f:
                f.write(f"{slide_path}\n")

    with open(os.path.join(args.logdir, f"DONE_split_{args.split_no}.txt"), "a") as f:
        f.write(f"{slide_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_no", type=int, required=False, default=0)
    parser.add_argument("--num_splits", type=int, required=False, default=1)
    parser.add_argument("--logdir", type=str, required=True)

    args = parser.parse_args()
    if args.split_no >= args.num_splits:
        raise ValueError("split_no must be < num_splits")
    main(args)




