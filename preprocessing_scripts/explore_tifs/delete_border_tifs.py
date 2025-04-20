import os
import re

# This is for looking at the tifs in QGIS and removing any border tifs that have black pixels as those are not relevant for this step in data preprocessing
tif_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\json_with_0\corresponding rgb masks"
json_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\COPY OF DATA\rgb rendered mask\json_with_0"

def should_delete(number):
    #Convert to integer
    num = int(number)
    
    # Check conditions:
    # 1. Numbers 1-24
    # 2. Numbers 624-648
    # 3. Numbers divisible by 24
    # 4. Numbers that give remainder 1 when divided by 24

    specific_numbers = {47, 599, 602, 614, 615, 616, 618, 619, 620, 621, 622, 623}


    if (1 <= num <= 24 or
        624 <= num <= 648 or
        num % 24 == 0 or
        num % 24 == 1) or num in specific_numbers:
        return True
    return False

def get_file_number(filename):
    numbers = re.findall(r'\d+', filename)
    return numbers[-1] if numbers else None

# Process TIF files
for tif_file in os.listdir(tif_dir):
    if tif_file.endswith('.tif'):
        number = get_file_number(tif_file)
        if number and should_delete(number):
            # Delete TIF file
            tif_path = os.path.join(tif_dir, tif_file)
            os.remove(tif_path)
            print(f"Deleted TIF: {tif_file}")
            
            # Find and delete corresponding JSON
            base_name = os.path.splitext(tif_file)[0]
            for json_file in os.listdir(json_dir):
                if json_file.endswith('.json') and get_file_number(json_file) == number:
                    json_path = os.path.join(json_dir, json_file)
                    os.remove(json_path)
                    print(f"Deleted JSON: {json_file}")
                    break

print("Processing complete!")
