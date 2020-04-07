nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

initiatedAtNoise(X, Y) :- initiatedAt(X, Y).

% Number of timestamps to look at (should be min the length of the sequence)
givenRemaining(5).

initiatedAt(sequence0 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([0, 0, 0, 0, 0], Remaining, T).
initiatedAt(sequence0 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([1, 1, 1, 1, 1], Remaining, T).

initiatedAt(sequence1 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([2, 2, 2, 2, 2], Remaining, T).
initiatedAt(sequence1 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([3, 3, 3, 3, 3], Remaining, T).

initiatedAt(sequence2 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([4, 4, 4, 4, 4], Remaining, T).
initiatedAt(sequence2 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([5, 5, 5, 5, 5], Remaining, T).

initiatedAt(sequence3 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([6, 6, 6, 6, 6], Remaining, T).
initiatedAt(sequence3 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([7, 7, 7, 7, 7], Remaining, T).

initiatedAt(sequence4 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([8, 8, 8, 8, 8], Remaining, T).
initiatedAt(sequence4 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([9, 9, 9, 9, 9], Remaining, T).

% An empty sequence will always be within
sequenceWithin([], _, _).

% A sequence can be within Remaining of T if it starts at T
sequenceWithin(L, Remaining, T) :-
    sequenceEndingAt(L, Remaining, T).

% A sequence can be within Remaining of T if it is within NextRemaining of Tprev
sequenceWithin(L, Remaining, T) :-
    Remaining > 0,
    T >= 0,
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    sequenceWithin(L, NextRemaining, Tprev).

% A sequence will start at T and be within Remaining if X happens at T and the rest of the sequence is within NextRemaining of Tprev
sequenceEndingAt([X | L], Remaining, T) :-
    Remaining > 0,
    T >= 0,
    happensAt(Y, T),
    digit(Y, X),
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    sequenceWithin(L, NextRemaining, Tprev).

sdFluent( aux ).
