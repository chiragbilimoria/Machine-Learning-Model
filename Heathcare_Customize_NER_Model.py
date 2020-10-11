# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

# Here I am writing a python code to identify medical code from given string. This case study is related to EHR(electronic health records). 
# Here in Patients records we get a string which contains medical codes and its description. 
# So I am building a custom NER (Name Entity Recognition) model which tells us what is code and what is description in given string
# NER helps us in finding named entities in the given string. For example India started manufacturing apple phones. So when we pass this string it will identify India as Geopolitical Entity and Apple are Org
# So with customizing NER module from en_core_web_sm library we can predict the what we want to capture from gievn string and use it according to our use case. I am using it for EHR use case
# Here usually the EHR data description is wrongly written and with identifying code 
# we can identify correct description for given medical code which helps us to determine what disease a patient is diagnosed with.
# This model predicts codes very well if code is either present in beginning or end of string

#importing libraries
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# %%
nlp=spacy.load('en_core_web_sm')


# getting pipes names
nlp.pipe_names


# Training model with actual medical data(code string)
train=[
        ('L02.31 Cutaneous abscess of buttock',{"entities":[(0,5,"CODE")]}),
        ('L08.9 Local infection of skin  subcutaneous tissue',{"entities":[(0,4,"CODE")]}),
        ('R30.0 Dysuria',{"entities":[(0,4,"CODE")]}),
        ('CHEST PAIN 786.50',{"entities":[(11,16,"CODE")]}),
        ('d50.9 iron deficiency anemia',{"entities":[(0,4,"CODE")]}),
        ('C54.1 Endometrial CA',{"entities":[(0,4,"CODE")]}),
        ('C62.11 Testicular CA',{"entities":[(0,5,"CODE")]}),
        ('C44.722 Squamous cell carcinoma of skin of right lower limb,including hip',{"entities":[(0,6,"CODE")]}),
        ('241571002 CT hip left wo contrast ',{"entities":[(0,8,"CODE")]}),
        ('PAIN IN LIMB 729.5',{"entities":[(13,17,"CODE")]}),
        ('Lymphoma DlBcl C83.35',{"entities":[(15,20,"CODE")]}),
        ('Pain lt lower limb M79.605',{"entities":[(19,25,"CODE")]}),
        ('Spinal stenosis lumbar region M48.062',{"entities":[(30,36,"CODE")]}),
        ('N13.2 Hydronephrosis with renal and ureteral calcu',{"entities":[(0,4,"CODE")]}),
        ('Rt shoulder dislocation S43.004A',{"entities":[(24,31,"CODE")]}),
        ('RHEUMATOID ARTHRITIS MULTIPLE SITES; M06.89',{"entities":[(37,42,"CODE")]}),
        ('241571999 CT hip left wo contrast SNOMED-CT',{"entities":[(0,8,"CODE")]}),
        ('241571888 CT hip left wo contrast SNOMED-CT',{"entities":[(0,8,"CODE")]})
        
]





# We want to modify(customize) ner pipe from 'en_core_web_sm' packages
ner=nlp.get_pipe("ner")

# Adding label from train data
for _,annotations in train:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Diabaling pipes/modules other then NER
disable_pipes=[pipe for pipe in nlp.pipe_names if pipe != 'ner']

# %%
import random
from spacy.util import minibatch,compounding
from pathlib import Path


# Traing our model in mini batches for 100 iteration and oupting the loss values
import random
from spacy.util import minibatch,compounding
from pathlib import Path

with nlp.disable_pipes(*disable_pipes):
    optimizer = nlp.resume_training()
    
    for iteration in range(100):
        
        random.shuffle(train)
        losses={}
        
        batches=minibatch(train)
        for batch in batches:
            text, annotation=zip(*batch)
            nlp.update(
                        text,
                        annotation,
                        drop=0.5,
                        losses=losses,
                        sgd=optimizer
                     )
            print("Losses",losses)


# Checking if our train data set was identify correclty with new entities that were added while training our model
for text,_ in train:
    doc=nlp(text)
    print('Entities',[(ent.text,ent.label_)for ent in doc.ents])

# Testing model with new string
code_string="Size greater than Dates Z36.88^^"

# Removing ^ which is the part of original data (In EHR data/medical data codes & description are separated by ^)
code_string=code_string.replace('^',"")

# %%
doc=nlp(code_string)

for ent in doc.ents:
    print(ent.text)

# model identifies code
code=ent.text

# printing code
code

# Extracting description after identifying codes
description = code_string.replace(code,'')

# printing description
description


