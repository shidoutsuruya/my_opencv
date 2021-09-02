from PIL import Image
import os.path
import glob

def jpg2pgm( jpg_file , pgm_dir,i ):
    jpg = Image.open( jpg_file )
    jpg = jpg.resize( (200,250) , Image.BILINEAR )
    name =(str)(os.path.join( pgm_dir , str(i)))+".pgm"
    jpg.save( name )
 
i=0
for jpg_file in glob.glob(r"C:\Users\max21\Desktop\Python\OpenCV\cars_train\*.jpg"):
    i=i+1
    jpg2pgm( jpg_file , r'C:\Users\max21\Desktop\Python\OpenCV\cars_train_pgm',i )
