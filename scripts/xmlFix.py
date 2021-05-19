import argparse
import os
import glob

def fix(closest_dir):
    og_dir = os.getcwd()
    print(f"[.] Changing wd: {closest_dir}")
    os.chdir(closest_dir)
    for xml in glob.glob('*.xml'):
        print(f"[.] Fixing {xml}")
        with open(xml, 'r') as f:
            tmp = f.read()
        with open(xml, 'w') as f:
            f.write('<data>')
            f.write(tmp)
            f.write('</data>')
    os.chdir(og_dir)
        
if __name__ == "__main__":
    folders = [
        'C:\\Users\\Matteo\\Documents\\tesi\\series_4',
        'C:\\Users\\Matteo\\Documents\\tesi\\series_5',
        'C:\\Users\\Matteo\\Documents\\tesi\\series_6',
        'C:\\Users\\Matteo\\Documents\\tesi\\series_8',
        'C:\\Users\\Matteo\\Documents\\tesi\\series_9',
    ]
    for d in folders:
        fix(d)