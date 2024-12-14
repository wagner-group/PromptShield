import numpy


b = numpy.load('result/attack_responses.npz')

#print(b)

a = b['data']
for item in a:
    print(item)

