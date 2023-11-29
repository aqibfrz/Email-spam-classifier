import wordcloud
text='hello()) how are you'
y = []
for i in text:
    if i.isalnum():
        y.append(i)

text=y[:]
print(text)    