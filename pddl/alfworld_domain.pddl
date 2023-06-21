(define (domain alfred)
    (:requirements :typing)

    (:types
        receptacle object rtype
    )

    (:constants
        SinkBasinType - rtype
        MicrowaveType - rtype
        FridgeType - rtype
    )

    (:predicates
        (atReceptacleLocation ?r - receptacle) ; true if the robot is at the receptacle location
        (inReceptacle ?o - object ?r - receptacle) ; true if object ?o is in receptacle ?r
        (openable ?r - receptacle) ; true if a receptacle is openable
        (opened ?r - receptacle) ; true if a receptacle is opened
        (isLight ?o - object) ; true if an object is light source
        (examined ?o - object ?l - object) ; whether the object has been looked at with light
        (holds ?o - object) ; object ?o is held by robot
        (isClean ?o - object) ; true if the object has been cleaned in sink
        (isHot ?o - object) ; true if the object has been heated up
        (isCool ?o - object) ; true if the object has been cooled
        (receptacleType ?r - receptacle ?t - rtype) ; true if the receptacle is of type ?t
    )

    ;; Examine an object (being held) using light source at location
    (:action examineObjectInLight
        :parameters (?o - object ?l - object ?r - receptacle)
        :precondition (and
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
        :parameters (?rEnd - receptacle)
        :effect (and
            (atReceptacleLocation ?rEnd)
            (forall
                (?r - receptacle)
                (when
                    (not (= ?r ?rEnd))
                    (not (atReceptacleLocation ?r))
                )
            )
        )
    )

    ; ;; robot opens receptacle
    (:action OpenReceptacle
        :parameters (?r - receptacle)
        :precondition (and
            (openable ?r)
            (atReceptacleLocation ?r)
            (not (opened ?r))
        )
        :effect (opened ?r)
    )

    ;; robot closes receptacle
    (:action CloseReceptacle
        :parameters (?r - receptacle)
        :precondition (and
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
        :parameters (?o - object ?r - receptacle)
        :precondition (and
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
        :parameters (?o - object ?r - receptacle)
        :precondition (and
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
        :parameters (?o - object ?r - receptacle)
        :precondition (and
            (receptacleType ?r SinkBasinType)
            (atReceptacleLocation ?r)
            (holds ?o)
        )
        :effect (isClean ?o)
    )

    ;; robot heats-up some object
    (:action HeatObject
        :parameters (?o - object ?r - receptacle)
        :precondition (and
            (receptacleType ?r MicrowaveType)
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
        :parameters (?o - object ?r - receptacle)
        :precondition (and
            (receptacleType ?r FridgeType)
            (atReceptacleLocation ?r)
            (holds ?o)
        )
        :effect (and
            (isCool ?o)
            (not (isHot ?o))
        )
    )
)