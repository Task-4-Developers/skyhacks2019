import jinja2

def html_generator():
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "test_1_brochure.html"
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(variable="xD")  # this is where to put args to the template renderer
    return outputText
if __name__ == "__main__":
    print(html_generator())
    