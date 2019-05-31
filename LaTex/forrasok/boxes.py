def boxes(imgpath):
    d = pytesseract.image_to_data(imgpath, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(imgpath, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("im", imgpath)
    cv2.imwrite(srcpath+"boxedimg.png",imgpath)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
