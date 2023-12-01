## Meaning of messages

|  Msg  |      Content      |
| :---: | :---------------: |
|   2   |       hint        |
|  10   |    battle cmd     |
|  11   |       idle        |
|  14   |   select option   |
|  15   |    select card    |
|  16   |   select chain    |
|  18   |   select place    |
|  20   |  select tribute   |
|  24   |   select place?   |
|  31   |   confirm card    |
|  33   |         -         |
|  50   |       move        |
|  54   |        set        |
|  60   |     summoning     |
|  61   |     summoned      |
|  62   | summoning special |
|  63   |         -         |
|  70   |     chaining      |
|  71   |         -         |
|  72   |         -         |
|  73   |         -         |
|  74   |         -         |
|  90   |       draw        |
|  91   |      damage       |
|  110  |      attack       |
|  111  |      battle       |
|  113  |                   |
|  114  |    end damage     |

## Typical workflow

**Summon Face-up Attack**

11 -> act_on_card -> 2 -> 18 -> 50 -> 60 -> [16 -> 16] -> 61 -> [16 -> 16] -> 11

**Summon Face-down Defense**

11 -> act_on_card -> 2 -> 18 -> 50 -> 54 -> [16 -> 16] -> 11

**Activate a Spell**

11 -> act_on_card -> 2 -> 18 -> 50 -> 70 -> 71 (Unhandled)

**Execute add card Effect**

50 -> 31 -> 33 -> 73 -> 50 -> 74 -> [16 -> 16] -> 11

**Set a Spell Face-down**

11 -> act_on_card -> 2 -> 18 -> 50 -> 54 -> [16 -> 16] -> 11

**Attack**

10 -> [2 -> 15] -> 110 -> 16 -> 16 -> 2 -> 16 -> 16 -> 113 -> [53] -> ... -> 91 -> [50] -> 114 -> 10
      ^                                                       ^                    ^
      with card                                               pos change           destroy

**Main1 -> End**

16 -> 41 -> [15] -> 40
            ^
            discard card

**Turn Start**

40 -> 41 -> 2? -> 90 -> [16 -> 16] -> 41 -> 41 -> 11

**Main1 -> Battle**

11 -> [16] -> 41 -> 10

**Standby**

12
