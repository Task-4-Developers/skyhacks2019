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


def to_pdf(htmlfilename):
    #windows
    #dont even think about changing this
    wkhtmltopdf_path="bonus_pdf\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
    wkhtmltopdf_path=os.path.abspath(wkhtmltopdf_path)
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    #generating pdf
    pdfkit.from_file(htmlfilename,htmlfilename.replace(".html",".pdf"),configuration=config)
    ###############

def generate_json(flat_name):
    nlp=NLP()
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

    with open("output_package.json","w") as output:
        flat_data=[] #whole flat
        for image_filename in flat_rooms:
            one_room_data=[] #single room
            single_element_data={     
                "adjective": "",
                "element": ""
            }   #single element

            #windows
            image_path=os.path.abspath("bonus_pdf\\main_task\\input\\%s\\%s"%(flat_name,image_filename))

            #json input file
            one_room_data.append(dict(link=image_path))
            with open(image_path+'.json', 'r') as f:
                room_details = json.load(f)

            #first element is a room
            single_element_data["adjective"]= nlp.give_adjective(room_details[0]['score'])
            single_element_data["element"]=room_details[0]['name']
            one_room_data.append(single_element_data)

            #this iterates all elements
            for distro in room_details[1:]:     
                single_element_data["adjective"]=nlp.give_adjective(distro['score'])
                single_element_data["element"]=distro['name']
                print(single_element_data["adjective"]+" "+single_element_data["element"])
                one_room_data.append(dict(single_element_data))

            flat_data.append(one_room_data)
        json.dump(flat_data,output)

if __name__ == "__main__":
    generate_json("test_1")