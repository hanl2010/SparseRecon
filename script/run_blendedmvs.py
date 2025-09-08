import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='BlendedMVS_PATH')
parser.add_argument('--conf', type=str, default='confs/blendedmvs.conf')
parser.add_argument("--feature_extractor", choices=["vismvsnet", "mvsformer", "transmvsnet"], default="vismvsnet")
args = parser.parse_args()


scenes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
patch_loss_weight = 0.5

excluded_gpus = set()
mem_threshold = 0.1


jobs = scenes

def train_block(gpu_id, scene):
       cmd = (
              f'CUDA_VISIBLE_DEVICES={gpu_id} python exp_runner.py --conf {args.conf} --case scan{scene} '
              f'--data_path {args.data_path} '
              f'--mode train '
              f'--use_rgb_pixel_loss '
              f'--use_rgb_patch_loss '
              f'--use_feat_loss '
              f'--use_depth_loss '
              f'--use_embed_mask '
              f'--use_occ_mask '
              f'--rgb_patch_loss_weight {patch_loss_weight} '
              f'--feature_extractor {args.feature_extractor} '
              # f'--is_continue '
              )
       print(cmd)
       os.system(cmd)

       cmd = (
              f'CUDA_VISIBLE_DEVICES={gpu_id} python exp_runner.py --conf {args.conf} --case scan{scene} '
              f'--data_path {args.data_path} '
              f'--mode validate_mesh '
              f'--is_continue '
              )
       print(cmd)
       os.system(cmd)

       cmd = (
              f'CUDA_VISIBLE_DEVICES={gpu_id} python script/clean_mesh_bmvs.py '
              f'--input_mesh output/blendedmvs/scan{scene}/meshes/00100000_512.ply '
              f'--scan_id {scene} '
              f'--mask_dir {args.data_path} '
              )
       print(cmd)
       os.system(cmd)

       return True

def worker(gpu_id, scene):
    print(f"Starting job on GPU {gpu_id} with scene {scene}\n")
    train_block(gpu_id, scene)
    print(f"Finished job on GPU {gpu_id} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
       future_to_job = {}
       reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

       while jobs or future_to_job:
              # Get the list of available GPUs, not including those that are reserved.
              all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxLoad=0.9, maxMemory=mem_threshold))
              available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

              # Launch new jobs on available GPUs
              while available_gpus and jobs:
                     gpu = available_gpus.pop(0)
                     job = jobs.pop(0)
                     future = executor.submit(worker, gpu, job)  # Unpacking job as arguments to worker
                     future_to_job[future] = (gpu, job)
                     reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

              # Check for completed jobs and remove them from the list of running jobs.
              # Also, release the GPUs they were using.
              done_futures = [future for future in future_to_job if future.done()]
              for future in done_futures:
                     job = future_to_job.pop(future)  # Remove the job associated with the completed future
                     gpu = job[0]  # The GPU is the first element in each job tuple
                     reserved_gpus.discard(gpu)  # Release this GPU
                     print(f"Job {job} has finished., rellasing GPU {gpu}")
              # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
              # when there are no GPUs available.
              if len(jobs) > 0:
                     print("No GPU available at the moment. Retrying in 1 minutes.")
                     time.sleep(60)
              else:
                     time.sleep(10)

       print("All blocks have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=10) as executor:
    dispatch_jobs(jobs, executor)
