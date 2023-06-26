(define (problem alf)
    (:domain alfred)
    (:objects
        bed-1 - bed
        desk-2 - desk
        desk-1 - desk
        drawer-6 - drawer
        drawer-5 - drawer
        drawer-4 - drawer
        drawer-3 - drawer
        drawer-2 - drawer
        drawer-1 - drawer
        garbagecan-1 - garbagecan
        laundryhamper-1 - laundryhamper
        safe-1 - safe
        shelf-6 - shelf
        shelf-5 - shelf
        shelf-4 - shelf
        shelf-3 - shelf
        shelf-2 - shelf
        shelf-1 - shelf
        book-1 - book
        desklamp-1 - desklamp
    )
    (:init
        (atReceptacleLocation desk-1)
        (isReceptacle bed-1)
        (isReceptacle desk-2)
        (isReceptacle desk-1)
        (isReceptacle drawer-6)
        (isReceptacle drawer-5)
        (isReceptacle drawer-4)
        (isReceptacle drawer-3)
        (isReceptacle drawer-2)
        (isReceptacle drawer-1)
        (isReceptacle garbagecan-1)
        (isReceptacle laundryhamper-1)
        (isReceptacle safe-1)
        (isReceptacle shelf-6)
        (isReceptacle shelf-5)
        (isReceptacle shelf-4)
        (isReceptacle shelf-3)
        (isReceptacle shelf-2)
        (isReceptacle shelf-1)
        (openable drawer-6)
        (opened drawer-6)
        (openable drawer-5)
        (opened drawer-5)
        (openable drawer-4)
        (opened drawer-4)
        (openable drawer-3)
        (opened drawer-3)
        (openable drawer-2)
        (opened drawer-2)
        (openable drawer-1)
        (opened drawer-1)
        (openable safe-1)
        (opened safe-1)
        (holds book-1)
        (isLight desklamp-1)
        (inReceptacle desklamp-1 desk-1)
    )
    (:goal
        (exists
            (?t - book ?l - desklamp)
            (examined ?t ?l)
        )
    )
)