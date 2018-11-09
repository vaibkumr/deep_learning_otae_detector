# Shimura tae detector
![](https://myanimelist.cdn-dena.com/images/characters/14/115884.jpg)
Shimura Tae (志村 妙 Shimura Tae), or just Otae (お妙) (the "お" (O) is an honorific used to refer to women), is the older sister of Shimura Shinpachi. She works at Snack Smile as a cabaret hostess to keep the family doujo Koudoukan going.

This is a deep learning model created to detect Otae san. I've heavily used the
fast.ai library and learned to make this in lesson 1 of fast.ai deep learning 1 course.
      
  
- Architecture</b>: resnet34 <br>
- Training data</b>: I collected by taking screenshots while watching the anime Gintama
- There were total 1600 images with 700 of them marked as "otae" for the training. 
- Request sent by: jquery and ajax 
- Backend: Flask
- Currently hosted [here](https://otae.netlify.com/) (only the frontend. I don't have money to buy a server. Pythonanywhere doesn't support pytorch for free servers)/
