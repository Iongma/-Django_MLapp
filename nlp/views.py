from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
import pickle

index2category = pd.read_csv("index2category.csv")["v"].to_dict()

with open("rf.pickle",mode="rb")as f:
    model = pickle.load(f)


def index(request):
    if request.method == "GET":
        return render(request, "nlp/home.html")

    else:
        inputed_value = [request.POST['inputed_value']]
        result = model.predict(inputed_value)[0]
        pred = index2category[result]

        return render(request,"nlp/home.html", {"pred":pred})
