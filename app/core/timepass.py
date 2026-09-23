a='this'
b='that'

if a is None:
    if b is None:
        print ("")
elif a or b is not None:
    if a and b is not None:
        print(a + b)
    elif a is None:
        print(b)
    elif b is None:
        print(a)

