import os

from problog.extern import problog_export


@problog_export('+str', '+term', '-str')
def open_(filename, term):
    # Remove the quotes
    filename = filename[1:-1]

    if os.path.isfile(filename):
        os.remove(filename)

    return filename


@problog_export('+str', '+str', '+list')
def format(filename, text, arguments):
    with open(filename, 'a') as f:
        f.write(text[1:-1].replace('~w', '{}').format(*arguments))

    with open(filename, 'r') as f:
        contents = f.read()

    return []


@problog_export('+str')
def nl(filename):
    with open(filename, 'a') as f:
        f.write('\n')

    return []


@problog_export('+str')
def close_(filename):
    return []


@problog_export('+list', '+list', '-list')
def append(l1, l2):
    return l1 + l2


@problog_export('+list', '-list')
def sort_no_duplicates(l):
    # Remove duplicates
    l = list(set(l))

    l.sort()

    return l


@problog_export('+int', '+list', '-int')
def previousTimeStamp(t, timestamps):
    last = []

    for i in timestamps:
        if i >= t:
            return last

        last = i

    return []


@problog_export('+int', '+list', '-int')
def nextTimeStamp(t, timestamps):
    for i in timestamps:
        if i > t:
            return i

    return []
