(define (domain alfred)
    (:predicates
        (isReceptacle ?o - object) ; true if the object is a receptacle
        (atReceptacleLocation ?r - object) ; true if the robot is at the receptacle location
        (inReceptacle ?o - object ?r - object) ; true if object ?o is in receptacle ?r
        (openable ?r - object) ; true if a receptacle is openable
        (opened ?r - object) ; true if a receptacle is opened
        (isLight ?o - object) ; true if an object is light source
        (examined ?o - object ?l - object) ; whether the object has been looked at with light
        (holds ?o - object) ; object ?o is held by robot
        (isClean ?o - object) ; true if the object has been cleaned in sink
        (isHot ?o - object) ; true if the object has been heated up
        (isCool ?o - object) ; true if the object has been cooled
        (isSink ?o - object) ; true if the object is a sink
        (isMicrowave ?o - object) ; true if the object is a microwave
        (isFridge ?o - object) ; true if the object is a fridge
    )

    ;; Examine an object (being held) using light source at location
    (:action examineObjectInLight
        :parameters (?o - object ?l - object ?r - object)
        :precondition (and
            (isReceptacle ?r)
            (isLight ?l) ; is light source
            (holds ?o) ; agent holds object
            (atReceptacleLocation ?r) ; agent is at receptacle
            (inReceptacle ?l ?r) ; light source is in receptacle
            (or
                (not (openable ?r)) ; receptacle is not openable
                (opened ?r) ; object is in receptacle and receptacle is open
            )
        )
        :effect (examined ?o ?l)
    )

    ;; robot goes to receptacle
    (:action GotoReceptacle
        :parameters (?rEnd - object)
        :precondition (isReceptacle ?rEnd)
        :effect (and
            (atReceptacleLocation ?rEnd)
            (forall
                (?r - object)
                (when
                    (and
                        (isReceptacle ?r)
                        (not (= ?r ?rEnd))
                    )
                    (not (atReceptacleLocation ?r))
                )
            )
        )
    )

    ; ;; robot opens receptacle
    (:action OpenReceptacle
        :parameters (?r - object)
        :precondition (and
            (isReceptacle ?r)
            (openable ?r)
            (atReceptacleLocation ?r)
            (not (opened ?r))
        )
        :effect (opened ?r)
    )

    ;; robot closes receptacle
    (:action CloseReceptacle
        :parameters (?r - object)
        :precondition (and
            (isReceptacle ?r)
            (openable ?r)
            (atReceptacleLocation ?r)
            (opened ?r)
        )
        :effect (and
            (not (opened ?r))
        )
    )

    ;; robot picks up  object  from a receptacle
    (:action PickupObjectFromReceptacle
        :parameters (?o - object ?r - object)
        :precondition (and
            (isReceptacle ?r)
            (atReceptacleLocation ?r) ; agent is at receptacle
            (inReceptacle ?o ?r) ; object is in/on receptacle
            (not (isLight ?o)) ; object is not light source
            (forall ; agent's hands are empty.
                (?t - object)
                (not (holds ?t))
            )
            (or
                (not (openable ?r)) ; receptacle is not openable
                (opened ?r) ; object is in receptacle and receptacle is open
            )
        )
        :effect (and
            (not (inReceptacle ?o ?r)) ; object is not in receptacle
            (holds ?o) ; agent holds object
        )
    )

    ;; robot puts down an object
    (:action PutObject
        :parameters (?o - object ?r - object)
        :precondition (and
            (isReceptacle ?r)
            (atReceptacleLocation ?r)
            (holds ?o)
            (or (not (openable ?r)) (opened ?r)) ; receptacle is opened if it is openable
        )
        :effect (and
            (inReceptacle ?o ?r)
            (not (holds ?o))
        )
    )

    ; ;; agent cleans some object
    (:action CleanObject
        :parameters (?o - object ?r - object)
        :precondition (and
            (isReceptacle ?r)
            (isSink ?r)
            (atReceptacleLocation ?r)
            (holds ?o)
        )
        :effect (isClean ?o)
    )

    ;; robot heats-up some object
    (:action HeatObject
        :parameters (?o - object ?r - object)
        :precondition (and
            (isReceptacle ?r)
            (isMicrowave ?r)
            (atReceptacleLocation ?r)
            (holds ?o)
        )
        :effect (and
            (isHot ?o)
            (not (isCool ?o))
        )
    )

    ;; robot cools some object
    (:action CoolObject
        :parameters (?o - object ?r - object)
        :precondition (and
            (isReceptacle ?r)
            (isFridge ?r)
            (atReceptacleLocation ?r)
            (holds ?o)
        )
        :effect (and
            (isCool ?o)
            (not (isHot ?o))
        )
    )
)