import pdfkit
import os
import json

class NLP:
    def __init__(self):
        self.adjectives=[("in need of repair", "used"),( "slightly used", "in good condition"), ("awesome" ,"great", "noice" , "lovely")]
        self.step=1/len(self.adjectives)
        self.adjectives_used=[0 for _ in range(len(self.adjectives))]
    def give_adjective(self,score):
        #TODO make proper exception handling
        which_adj=int(min(score/self.step,len(self.adjectives)-1)) #not bigger than array
        self.adjectives_used[which_adj]+=1
        return (self.adjectives[which_adj][self.adjectives_used[which_adj]%len(self.adjectives[which_adj])])



#TODO generate propper html




def generate_html(flat_name):
    nlp=NLP()
    filename="bonus_pdf/%s_brochure.html"%flat_name
    
    #this finds all .jpg and puts them into flat_rooms
    flat_path="bonus_pdf\\main_task\\input\\%s\\"%flat_name
    flat_rooms=[]
    for _,_,f in os.walk(flat_path):
        for file in f:
            if ".jpg" in file:
                if ".json" in file: 
                    continue
                flat_rooms.append(file)
                # print(file)
    
    with open(filename,"w") as brochure:

        #generate html
        brochure.write("<!DOCTYPE html>")

        #CSS here
        brochure.write("<head><style>")
        with open("bonus_pdf/brochure_style.css") as css_style:
            for lines in css_style.readlines():
                brochure.write(lines)
        brochure.write("</style><head>")

        brochure.write("<body>")

        brochure.write("<div class=\"flat-header\">\n"
            + "<h1> %s </h1></div>\n" % flat_name)
        for image_filename in flat_rooms:
            # print(image_filename)
            image_path=os.path.abspath("bonus_pdf\\main_task\\input\\%s\\%s"%(flat_name,image_filename))
            brochure.write("<div class=\"grid-container\">\n")
            brochure.write("<div class=\"grid-item\"><img src=\"%s\" height=\"%s\" weight=\"%s\" /></div>" % (image_path,300,300)+"\n")
            #process json
            with open(image_path+'.json', 'r') as f:
                distros_dict = json.load(f)
            #this prints one room
            #Text begins here: room header
            brochure.write("<div class=\"grid-item\">")
            brochure.write("<h1> This %s %s contains:</h1>\n" %( nlp.give_adjective(distros_dict[0]['score']),distros_dict[0]['name']))
            brochure.write("<ul>")
            for distro in distros_dict[1:]:
                brochure.write("<li>"+ nlp.give_adjective(distro["score"]) +" "+distro['name']+"</li>"+"\n")
                print(distro['name'])
            brochure.write("</ul></div>")
            brochure.write("</div>\n")
        brochure.write("</body>")
    to_pdf(filename)

def to_pdf(htmlfilename):
    #dont even think about changing this
    wkhtmltopdf_path="bonus_pdf\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
    wkhtmltopdf_path=os.path.abspath(wkhtmltopdf_path)
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    #generating pdf
    pdfkit.from_file(htmlfilename,htmlfilename.replace(".html",".pdf"),configuration=config)

generate_html("test_1")

# to_pdf(filename)