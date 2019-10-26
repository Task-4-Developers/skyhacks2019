import pdfkit
import os
import json

#TODO generate propper html

filename="bonus_pdf/brochure.html"
brochure=open(filename,"w")

#generate html
brochure.write("<!DOCTYPE html>")
image_filename="0A36B673BA6513F772FB78FF597BE44F7E639A0F.jpg"
image_path=os.path.abspath("bonus_pdf\\main_task\\input\\test_1\\%s"%image_filename)
brochure.write("<img src=\"%s\" height=\"%s\" weight=\"%s\" />" % (image_path,300,300))

#process json
with open(image_path+'.json', 'r') as f:
    distros_dict = json.load(f)

brochure.write("<div class=\"content\">")
brochure.write("<h1> This %s contains:</h1>" % distros_dict[0]['name'])
brochure.write("<ul>")
for distro in distros_dict[1:]:
    brochure.write("<li>"+distro['name']+"</li>")
    print(distro['name'])
brochure.write("</ul>")
brochure.write("</div>")

brochure.close()

def to_pdf(htmlfilename):
    #dont even think about changing this
    wkhtmltopdf_path="bonus_pdf\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
    wkhtmltopdf_path=os.path.abspath(wkhtmltopdf_path)
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    #generating pdf
    pdfkit.from_file(htmlfilename,htmlfilename.replace(".html",".pdf"),configuration=config)

to_pdf(filename)