from sys import argv
from os import listdir
from os.path import isdir


def create_bone_file(root_folder):
    folders = listdir(root_folder)
   
    for folder in folders:
        print(f'Parsing {folder}...', end='')
        folder_path = f'{root_folder}/{folder}/'

        if not isdir(folder_path):
            continue
        
        f = open(folder_path + '/numbers.txt', 'r')
        bone_data = ' '.join([i.split()[-1] for i in f if i.strip() != ''])
        f.close()
        
        with open(f'{folder_path}/{folder}.bones', 'w') as f:
            f.write(bone_data)
        print('Done!')


def main():
    folders = argv[1:]
    for folder in folders:
        create_bone_file(folder)


if __name__ == '__main__':
    main()
