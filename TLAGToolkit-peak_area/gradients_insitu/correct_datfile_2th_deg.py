import os

# Set the path to the folder containing your .dat files
folder_path = 'C:/Users/eghiara/Desktop/ALBA/NCD_Mar24/DATA/PROCESSED/Y12095/007/'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace 'qn' with '2th_deg'
        updated_content = content.replace('q_nm^-1', '2th_deg')
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

print("Replacement completed.")