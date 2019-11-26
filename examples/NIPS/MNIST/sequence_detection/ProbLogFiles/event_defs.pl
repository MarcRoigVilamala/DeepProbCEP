nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

initiatedAt(sequence = true, T) :-
    happensAt(X, T),
    digit(X, 1),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 1).

initiatedAt(sequence = false, T) :-
    happensAt(X, T),
    digit(X, 0),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 0).

sdFluent( aux ).
