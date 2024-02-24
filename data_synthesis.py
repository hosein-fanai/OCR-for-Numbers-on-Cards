from PIL import Image, ImageDraw, ImageFont

import multiprocessing

import os

import random

from constants import input_shape, trainset_path, nums_per, nums_en, template_path, num_processes, generating_index

from utilities import normalize_bbox


def generate_fake_card(template_path, template_type, info, index):
    card_template = Image.open(template_path).convert("RGB").resize(input_shape[1::-1])
    draw = ImageDraw.Draw(card_template)

    font_color = (0, 0, 0)

    if template_type == 0:
        card_number, cvv2, exp_date = info 

        font_type = ["arial.ttf", "calibri.ttf", "times.ttf"]
        random.shuffle(font_type)
        
        card_number_font_size = random.randint(23, 40)
        exp_date_font_size = random.randint(10, 23)
            
        card_number_font = ImageFont.truetype(font_type[0], card_number_font_size)
        exp_date_font = ImageFont.truetype(font_type[0], exp_date_font_size)
    
        card_number_font = ImageFont.truetype(font_type[0], card_number_font_size)
        exp_date_font = ImageFont.truetype(font_type[0], exp_date_font_size)
    
        card_number_position = (random.randint(0, int(0.1*input_shape[1])), 
                                random.randint(int(0.35*input_shape[0]), int(0.75*input_shape[0])))
        cvv2_position = (random.randint(int(0.03*input_shape[1]), int(0.93*input_shape[1])), 
                        random.randint(int(0.05*input_shape[0]), int(0.93*input_shape[0])))
        exp_date_position = (random.randint(0, int(0.93*input_shape[1])), 
                            random.randint(0*input_shape[0], int(0.93*input_shape[0])))
    
        bboxes = []
        nums = []
        for char in card_number:
            draw.text(card_number_position, char, font=card_number_font, fill=font_color)
            
            x, y = card_number_position
            w, h = card_number_font.getbbox(char)[2:]
            
            card_number_position = (x+w, y)
            if char != ' ':
                bboxes.append((x, y, w, h))
                nums.append(int(char))
        
        draw.text(cvv2_position, cvv2, font=exp_date_font, fill=font_color)
        draw.text(exp_date_position, exp_date, font=exp_date_font, fill=font_color)

    elif template_type == 1:
        ID, name, lname, b_date, fname, exp_date = info 
                    
        font = ImageFont.truetype("calibri.ttf", 14)
    
        ID_position = (0.58*input_shape[1], 0.25*input_shape[0])
        name_position = (0.68*input_shape[1], 0.35*input_shape[0])
        lname_position = (0.59*input_shape[1], 0.45*input_shape[0])
        b_date_position = (0.59*input_shape[1], 0.56*input_shape[0])
        fname_position = (0.68*input_shape[1], 0.65*input_shape[0])
        exp_date_position = (0.59*input_shape[1], 0.75*input_shape[0])
    
        bboxes = []
        nums = []
        for char in ID:
            draw.text(ID_position, char, font=font, fill=font_color)
            x, y = ID_position
            w, h = font.getbbox(char)[2:]
            ID_position = (x+w, y)

            bboxes.append((x, y, w, h))
            nums.append(int(char))
        
        draw.text(name_position, name, font=font, fill=font_color)
        draw.text(lname_position, lname, font=font, fill=font_color)
        draw.text(b_date_position, b_date, font=font, fill=font_color)
        draw.text(fname_position, fname, font=font, fill=font_color)
        draw.text(exp_date_position, exp_date, font=font, fill=font_color)

    card_template.save(os.path.join(trainset_path, "images", f"{index}.png"), "PNG")
    with open(os.path.join(trainset_path, "annotations", f"{index}.txt"), 'w') as file:
        if template_type == 0:
            cvv2 = cvv2.replace('CVV2: ', '')
            exp_date = exp_date.replace('/', '').replace('انقضا: ', '')
            if cvv2[-1] in nums_per:
                cvv2 = "".join([nums_en[nums_per.index(num)] for num in cvv2])
                exp_date = "".join([nums_en[nums_per.index(num)] for num in exp_date])
            file.write(f"{template_type} {cvv2} {exp_date}\n")
        elif template_type == 1:
            exp_date = "".join([nums_en[nums_per.index(num)] for num in exp_date.replace('/', '')])
            file.write(f"{template_type} 0 {exp_date}\n")

        for bbox, num in zip(bboxes, nums):
            if num in nums_per:
                num = nums_en[nums_per.index(num)]

            bbox = normalize_bbox(bbox)
            file.write(f"{num} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def generate_fake_credit_info(lang="english"):
    if lang == "english":
        chars = nums_en.copy()
    elif lang == "persian":
        chars = nums_per.copy()

    space = random.randint(1, 3)

    card_number = ""
    for i in range(16):
        if i % 4 == 0:
            card_number += ' ' * space
        random.shuffle(chars)
        card_number += str(chars[0])

    cvv2 = "CVV2: "
    for i in range(4):
        random.shuffle(chars)
        cvv2 += str(chars[0])

    exp_date = "انقضا: "
    for i in range(4):
        if i == 2:
            exp_date += '/'
        random.shuffle(chars)
        exp_date += str(chars[0])
    
    return card_number.lstrip(), cvv2, exp_date


def generate_fake_ID_info():
    nums = nums_per.copy()
    chars = list("صثقفعهخحجچپمنالبیسشطزردوکگءغضئآذژظ")
    
    ID = ""
    for _ in range(10):
        random.shuffle(nums)
        ID += str(nums[0])

    name = ""
    for _ in range(random.randint(3, 7)):
        random.shuffle(chars)
        name += str(chars[0])

    lname = ""
    for _ in range(random.randint(4, 10)):
        random.shuffle(chars)
        lname += str(chars[0])

    b_date = ""
    for i in range(8):
        if i in (4, 6):
            b_date += '/'
        random.shuffle(nums)
        b_date += str(nums[0])

    fname = ""
    for _ in range(random.randint(3, 7)):
        random.shuffle(chars)
        fname += str(chars[0])

    exp_date = ""
    for i in range(8):
        if i in (4, 6):
            exp_date += '/'
        random.shuffle(nums)
        exp_date += str(nums[0])
    
    return ID, name, lname, b_date, fname, exp_date


def generate_image(temp_full_path, template_type, index):
    print(f"in the generate_image function number {index}")
    if template_type == 0:
        info = generate_fake_credit_info(lang="english")
    elif template_type == 0.5:
        info = generate_fake_credit_info(lang="persian")
    elif template_type == 1:
        info = generate_fake_ID_info()
        
    generate_fake_card(
        temp_full_path, 
        template_type=int(template_type), 
        info=info,
        index=index
    )


def generate_multiple_images(temp_path, num_steps, gene_type, print_message="Generating images"):
    global generating_index

    print(f"{print_message} from {temp_path} ...")
    temp_full_path = os.path.join(template_path, "national ID" if gene_type == 1 else "credit", temp_path)

    pool = multiprocessing.Pool(processes=num_processes)
    pool.starmap(
        generate_image, 
        zip([temp_full_path]*num_steps, 
            [gene_type]*num_steps, 
            range(generating_index, generating_index+num_steps))
    )
    pool.close()
    pool.join()

    generating_index += num_steps
    print(f"The last generating index is: {generating_index-1}")


