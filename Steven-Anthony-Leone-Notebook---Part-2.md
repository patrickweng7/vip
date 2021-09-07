# Team Member
<b>Team Member:</b> Steven Leone <br>
<b> Major: </b> Computer Science <br>
<b>Email:  </b>sleone6@gatech.edu <br>
<b>Cell Phone:</b> (412)-378-7253 <br>
<b>Interests:</b> Machine Learning, Natural Language Processing, Software Engineering, Algorithms <br>
<b>Sub Team:</b> NLP

# Fall 2021

## Week 1
* General Meeting Notes
** We discussed potential sub team ideas
**A brainstorming channel in the slack was created for the NLP sub team.
** I think exploring how AutoML can be used to explore more complex NLP problems, like machine translation, would be interesting, especially as machine translations require multiple qualities to score, making it a natural choice for a multi objective framework.
**I found a few papers on using AutoML for machine translation. They each express how AutoML hasn't been used much for machine translation, and neither of them used multiple objectives (both used BLEU).
*** https://ieeexplore.ieee.org/abstract/document/9095246/ 
*** http://proceedings.mlr.press/v97/so19a.html
** We held a brainstorming meeting. We decided that the issue of complexity was best left to the NAS team if we were splitting this semester.

{| class="wikitable"
!Index
!Error Title
!Cause of Error
!How to Resolve
|-
|1
|Server won't start
|Port is likely in use via submitted job or terminal
|qstat or lsof -i:Port# , then “qdel ID” or “kill Port#” (respectively)
|}

## Week 2
* General Meeting Notes
** During the general meeting, I informed the whole team of ideas discussed in our brainstorm meeting. Devan also suggested added more primitives for more than embeddings.
** The sub team for NLP was officially formed with members decided.