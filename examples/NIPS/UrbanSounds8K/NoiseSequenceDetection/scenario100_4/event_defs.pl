nn(sound_net,[X],Y,[air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music]) :: sound(X,Y).

initiatedAtNoise(X, Y) :- initiatedAt(X, Y).

% Number of timestamps to look at (should be min the length of the sequence)
givenRemaining(4).

initiatedAt(sequence0 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([air_conditioner, air_conditioner], Remaining, T).
initiatedAt(sequence0 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([car_horn, car_horn], Remaining, T).

initiatedAt(sequence1 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([children_playing, children_playing], Remaining, T).
initiatedAt(sequence1 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([dog_bark, dog_bark], Remaining, T).

initiatedAt(sequence2 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([drilling, drilling], Remaining, T).
initiatedAt(sequence2 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([engine_idling, engine_idling], Remaining, T).

initiatedAt(sequence3 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([gun_shot, gun_shot], Remaining, T).
initiatedAt(sequence3 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([jackhammer, jackhammer], Remaining, T).

initiatedAt(sequence4 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([siren, siren], Remaining, T).
initiatedAt(sequence4 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([street_music, street_music], Remaining, T).

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
    sound(Y, X),
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    sequenceWithin(L, NextRemaining, Tprev).

sdFluent( aux ).
