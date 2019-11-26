nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

t(0.5)::initiatedAt(sequence0 = true, T) :-
    happensAt(X, T),
    digit(X, 0),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 0).

t(0.5)::initiatedAt(sequence0 = true, T) :-
    happensAt(X, T),
    digit(X, 1),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    previousTimeStamp(Tprev, Timestamps, Tprevprev),
    Tprevprev >= 0,
    happensAt(Xprev, Tprevprev),
    digit(Xprev, 1).

initiatedAt(sequence0 = false, T) :-
    happensAt(X, T),
    digit(X, 1),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 1).

initiatedAt(sequence1 = true, T) :-
    happensAt(X, T),
    digit(X, 2),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 2).

initiatedAt(sequence1 = false, T) :-
    happensAt(X, T),
    digit(X, 3),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 3).

initiatedAt(sequence2 = true, T) :-
    happensAt(X, T),
    digit(X, 4),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 4).

initiatedAt(sequence2 = false, T) :-
    happensAt(X, T),
    digit(X, 5),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 5).

initiatedAt(sequence3 = true, T) :-
    happensAt(X, T),
    digit(X, 6),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 6).

initiatedAt(sequence3 = false, T) :-
    happensAt(X, T),
    digit(X, 7),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 7).

initiatedAt(sequence4 = true, T) :-
    happensAt(X, T),
    digit(X, 8),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 8).

initiatedAt(sequence4 = false, T) :-
    happensAt(X, T),
    digit(X, 9),
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    Tprev >= 0,
    happensAt(Xprev, Tprev),
    digit(Xprev, 9).

sdFluent( aux ).
