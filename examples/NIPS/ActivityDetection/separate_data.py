if __name__ == '__main__':
    with open('/home/marc/projectes/deepproblog/examples/NIPS/ActivityDetection/trainlist02.txt', 'r') as f:
        training_videos = [l.strip() for l in f]

    with open('/home/marc/projectes/deepproblog/examples/NIPS/ActivityDetection/annotated_data.txt', 'r') as f:
        training = []
        testing = []

        for l in f:
            for t in training_videos:
                if t in l:
                    training.append(l)
                    break
            else:
                testing.append(l)

    something = 0
    for t in training:
        if 'something' in t:
            something += 1

    total_training = len(training)

    print(
        'Of the {} traning cases, {} are "something" and {} are "nothing"'.format(
            total_training,
            something,
            total_training - something
        )
    )

    # with open('training01.txt', 'w') as f:
    #     f.writelines(training)
    #
    # with open('testing01.txt', 'w') as f:
    #     f.writelines(testing)
