nn(sound_net,[X],Y,[start, other]) :: shots(X,Y).

initiatedAtNoise(X, Y) :- initiatedAt(X, Y).

% Number of timestamps to look at (should be min the length of the sequence)
givenRemaining(2).

initiatedAt(sequence0 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([start, start], Remaining, T).

initiatedAt(sequence0 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([other, other], Remaining, T).

%initiatedAt(sequence0 = false, T) :-
%    givenRemaining(Remaining),
%    \+ appearsIn(start, Remaining, T).

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
    shots(Y, X),
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    sequenceWithin(L, NextRemaining, Tprev).

% C will appear in Remaining of T if it happens at T
appearsIn(C, Remaining, T) :-
    Remaining > 0,
    T >= 0,
    happensAt(Y, T),
    shots(Y, C).

% C can be within Remaining of T if it appears in NextRemaining of Tprev
appearsIn(C, Remaining, T) :-
    Remaining > 0,
    T >= 0,
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    appearsIn(C, NextRemaining, Tprev).

sdFluent( aux ).
