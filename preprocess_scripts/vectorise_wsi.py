import os
import sys
from os.path import isfile, join
from os import listdir
from pathlib import Path
import multiprocessing

sys.path.append('../')
sys.path.append('../../')

from preprocess_utils.vectorize_wsi import SlideIterator


def multi_process_tasks(todo_tasks, function_to_run, k_args=[]):
    # set how many processes to use automatically
    # num_processes = multiprocessing.cpu_count()

    # set how many processes to use manually
    num_processes = 5

    pool = multiprocessing.Pool(num_processes)

    # comment this out
    converted_list = [f for f in listdir(vector_path) if "vectorised.txt" in listdir(join(vector_path, f))]

    converted_images = [f.split('.')[0] for f in converted_list]

    final_list = []

    for img in todo_tasks:
        if img.split('.')[0] not in converted_images:
            final_list.append(img)

    todo_tasks = final_list
    # until here

    num_tasks = len(todo_tasks)

    if num_tasks > 0:
        if num_tasks != 0:
            if num_processes > num_tasks:
                num_processes = num_tasks
            images_per_process = num_tasks / num_processes

            print("Number of processes: " + str(num_processes))
            print("Number of training images: " + str(num_tasks))

            # each task specifies a range of slides
            tasks = []
            for num_process in range(1, num_processes + 1):

                start_index = (num_process - 1) * images_per_process
                end_index = num_process * images_per_process

                start_index = int(start_index)
                end_index = int(end_index)

                tasks.append(todo_tasks[start_index: end_index])

                if start_index == end_index:
                    print("Task #" + str(num_process) + ": Process slide " + str(start_index))
                else:
                    print(
                        "Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

            # start tasks
            results = []
            for t in tasks:
                results.append(pool.apply_async(function_to_run, args=[t, *k_args]))

            count = 0
            for result in results:
                count += result.get()
                if count == num_tasks:
                    print("\nDone processing %d tasks" % count)


def vectorize_slides(slide_list, impatch_size):
    for slide in slide_list:
        slide_directory = join(vector_path, slide)
        Path(slide_directory).mkdir(parents=True, exist_ok=True)

        if not isfile(join(slide_directory, 'vectorised.npy')):
            print(f'Processing {slide_directory}')
            f = open(join(slide_directory, "vectorised.txt"), "w")
            f.close()

            try:
                # Read slide
                si = SlideIterator(
                    image_path=join(original_slides, slide),
                    threshold_mask=16,
                )

                # Process it
                si.save_array(
                    patch_size=impatch_size,
                    stride=impatch_size,
                    output_pattern=slide_directory + '/{item}',
                    downsample=1
                )
            except:
                print("*** ERROR processing %s slide ***" % slide)
                os.remove(join(slide_directory, "vectorised.txt"))

        if isfile(join(slide_directory, 'patches.npy')):
            f = open(join(slide_directory, "vectorised.txt"), "w")
            f.write("Done")
            f.close()


    return len(slide_list)


if __name__ == "__main__":
    impatch_size = 256
    original_slides = sys.argv[1]
    vector_path = sys.argv[2]
    extension = sys.argv[3]

    Path(vector_path).mkdir(parents=True, exist_ok=True)

    all_images = [f for f in listdir(original_slides) if (isfile(join(original_slides, f)) and f.endswith(extension))]

    multi_process_tasks(all_images, vectorize_slides, k_args=[impatch_size])
