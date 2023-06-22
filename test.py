from text2tags import TaggerLlama

model = TaggerLlama()

tags = model.predict_tags("An elven girl with pointed ears and silver hair holds up a plate of onigiri while smiling at the viewer from inside a room illuminated by sunlight coming through a window. She has bags under her eyes and her bangs cover them partially. Her hair is adorned with a hair flower")
print(tags)