while True:
    try:
        x = int(input('number please:'))
    except ValueError:
        print('not number')