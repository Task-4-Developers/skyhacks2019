import jinja2

import lobbed_bonus_package_linux as lbp

def html_generator():
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "template.html"


    flat_data = lbp.generate_json("test_1")
    
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(flat_name="The greatest place to live", flat_data=flat_data)  # this is where to put args to the template renderer
    return outputText
if __name__ == "__main__":
    with open("final.html", "w") as file:
        file.write(html_generator()) 

    lbp.to_pdf("final.html")

    # print(html_generator())
    