import pdfkit
import os

#TODO generate propper html

filename="brochure.html"
brochure=open(filename,"w")

#generate html
brochure.write("<!DOCTYPE html>")
image_filename="0A36B673BA6513F772FB78FF597BE44F7E639A0F.jpg"
image_path=os.path.abspath("bonus_pdf\\main_task\\input\\test_1\\%s"%image_filename)
brochure.write("<img src=\"%s\" height=\"%s\" weight=\"%s\" />" % (image_path,300,300))
brochure.write("<p> Andrzej </p>")

brochure.close()

#dont even think about changing this
wkhtmltopdf_path="bonus_pdf\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
wkhtmltopdf_path=os.path.abspath(wkhtmltopdf_path)
config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
#generating pdf
pdfkit.from_file(filename,filename.replace(".html",".pdf"),configuration=config)
