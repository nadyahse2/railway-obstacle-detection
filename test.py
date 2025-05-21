import subprocess
import os
import glob
import re
import sys
def run_test(image_path):
    try:
        command = [sys.executable, 'road.py', image_path]
        process = subprocess.run(command, capture_output=True,text=True,encoding='utf-8')
        output = process.stdout.strip()
        print(output)
        match = re.search(r"(Завал (есть|нет))",output)
        return match.group(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running script:{e}")
        return 0


zaval_folder = "test_images_with_zaval"
no_zaval_folder = "test_images_without_zaval"
zaval_images = glob.glob(os.path.join(zaval_folder, "*.jpg")) + glob.glob(os.path.join(zaval_folder, "*.png"))+glob.glob(os.path.join(zaval_folder, "*.PNG"))
no_zaval_images = glob.glob(os.path.join(no_zaval_folder, "*.jpg")) + glob.glob(os.path.join(no_zaval_folder, "*.png"))+glob.glob(os.path.join(no_zaval_folder, "*.PNG"))
count_failed_tests =0
print("Testing images with zavals:")
for image_path in zaval_images:
   image_name = os.path.basename(image_path)
   actual_result = run_test(image_path)
   expected_result = "Завал есть"
   if actual_result == expected_result:
       print(f"Test passed for {image_name}: Expected '{expected_result}', got '{actual_result}'")
   else:
       count_failed_tests+=1
       print(f"Test failed for {image_name}: Expected '{expected_result}', got '{actual_result}'")
print("Testing images without zavals:")
for image_path in no_zaval_images:
   image_name = os.path.basename(image_path)
   actual_result = run_test(image_path)
   expected_result = "Завал есть"
   if actual_result == expected_result:
       print(f"Test passed for {image_name}: Expected '{expected_result}', got '{actual_result}'")
   else:
       count_failed_tests+=1
       print(f"Test failed for {image_name}: Expected '{expected_result}', got '{actual_result}'")
print(count_failed_tests)
