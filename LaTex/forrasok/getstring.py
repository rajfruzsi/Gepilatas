def getstring():
    config= ("-psm 7")
    result=pytesseract.image_to_string(Image.open(srcpath + "thresh.png"),config=config)
    file=open("result", "w")
    file.write(result)
    return result
