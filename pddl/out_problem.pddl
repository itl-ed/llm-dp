(define (problem alf)
(:domain alfred)
(:objects cabinet-6 - receptacle
cabinet-5 - receptacle
cabinet-4 - receptacle
cabinet-3 - receptacle
cabinet-2 - receptacle
cabinet-1 - receptacle
coffeemachine-1 - receptacle
countertop-3 - receptacle
countertop-2 - receptacle
countertop-1 - receptacle
drawer-3 - receptacle
drawer-2 - receptacle
drawer-1 - receptacle
fridge-1 - receptacle
garbagecan-1 - receptacle
microwave-1 - receptacle
shelf-3 - receptacle
shelf-2 - receptacle
shelf-1 - receptacle
sinkbasin-1 - receptacle
stoveburner-4 - receptacle
stoveburner-3 - receptacle
stoveburner-2 - receptacle
stoveburner-1 - receptacle
toaster-1 - receptacle
lettuce-1 - object
)
(:init (openable cabinet-6)
(openable cabinet-5)
(openable cabinet-4)
(openable cabinet-3)
(openable cabinet-2)
(openable cabinet-1)
(openable drawer-3)
(openable drawer-2)
(openable drawer-1)
(receptacleType fridge-1 FridgeType)
(openable fridge-1)
(atReceptacleLocation fridge-1)
(opened fridge-1)
(receptacleType microwave-1 MicrowaveType)
(openable microwave-1)
(receptacleType sinkbasin-1 SinkBasinType)
(inReceptacle lettuce-1 countertop-1)
)
(:goal (exists (?t - object)
(and (inReceptacle ?t countertop-1)
(isCool ?t)
))))