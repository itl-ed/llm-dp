(define (problem alf)
(:domain alfred)
(:objects bed-1 - receptacle
desk-2 - receptacle
desk-1 - receptacle
drawer-6 - receptacle
drawer-5 - receptacle
drawer-4 - receptacle
drawer-3 - receptacle
drawer-2 - receptacle
drawer-1 - receptacle
garbagecan-1 - receptacle
laundryhamper-1 - receptacle
safe-1 - receptacle
shelf-6 - receptacle
shelf-5 - receptacle
shelf-4 - receptacle
shelf-3 - receptacle
shelf-2 - receptacle
shelf-1 - receptacle
book-1 - object
desklamp-1 - object
)
(:init (atReceptacleLocation desk-1)
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
(:goal (exists (?t - object ?l - object) 
(examined ?t ?l)
)))