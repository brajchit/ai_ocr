from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .models import Greeting

# Create your views here.
def index(request):
    # return HttpResponse('Hello from Python!')
    print("coming image file")
    if request.method == 'POST' and request.FILES['myfile']:

        #Archivo subido desde el front
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filenameExt = myfile.name
        filenameExt = filenameExt.split('.')[-1]

        #nombre del Archivo que se rescribe con el mismo nombra cada vez que sube una nueva letra
        # uniqueFileName = 'charTarget.'+filenameExt
        uniqueFileName = 'charTarget.jpg'

        fs.delete( uniqueFileName )
        filename = fs.save( uniqueFileName, myfile)

        uploaded_file_url = fs.url(filename)
        return render(request, 'index.html', {
            'uploaded_file_url': uploaded_file_url,
            'charFounded': True,
            ##Aqui va la letra que es reconocida ************************************
            'character': 'X'
        })
    # return render(request, 'core/simple_upload.html')
    return render(request, 'index.html')


def db(request):

    greeting = Greeting()
    greeting.save()

    greetings = Greeting.objects.all()

    return render(request, 'db.html', {'greetings': greetings})
