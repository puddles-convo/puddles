---
name: grade-support-conversations
description: How to rate an emotional-support conversation on empathy, relevance and distress, and label each helper turn, the way the people who sought help rated theirs. Read it before every rating.
source: ESConv (thu-coai/esconv, revision 32c54c15e7651cd68dfab18eee39f63ff0cc1e82), training split only
seed: 0
builder_model: Claude Opus 5.5 (claude-opus-5-5), written in a Claude Code session from the builder's ESConv training-split samples (seed 0)
---

# Grading support conversations

This guide was built from real support conversations and the ratings the people seeking help gave them afterwards. Use it to place a new conversation on the 1 to 5 scales. The rules in your instructions still apply; this guide only makes them concrete.

How to use it:
- Read the whole conversation first, then compare it with the anchors and examples below.
- 3 is an ordinary, acceptable conversation. Give 1 or 5 only when the conversation clearly matches that anchor.
- The examples show the seekers' own ratings. Rate a new conversation on what it shows, not by its similarity to an example.

## Empathy: did the helper show it understood how the person felt?

- **1:** The person's feelings are ignored, dismissed or met with cheerleading; the replies would read the same for anyone.
- **3:** Feelings are acknowledged in general terms ("sorry to hear that"), but the helper moves on to questions or advice without showing it understood this person's feeling.
- **5:** The helper repeatedly shows it understood this person's specific feelings, responds to what matters most to them, and the person seems to feel heard.

Raises the score:
- Names or acknowledges the specific feeling the person stated, and stays with it before moving on.
- Invites the person to say more about how they feel, and follows up on their answer instead of changing the subject.
- Responds directly to a serious disclosure: feeling worthless, being unsafe, not coping.
- Offers help in the form the person asked for: listens when they want to talk, gives ideas when they ask for them.
- Stays on the person's side without judging or correcting them.

Lowers the score:
- Answers a stated feeling with cheerfulness, good news or a silver lining instead of acknowledging it.
- Uses reassurance that fits anyone ("everyone feels that way", "you are not alone") in place of engaging with this person.
- Blames or lectures the person for their situation.
- Turns the attention to the helper: long stories about themselves, small talk about their own day.
- Passes over a serious disclosure with one line and a change of subject.
- Gives curt or mechanical replies right after the person shares something painful.

Traps:
- People rated most of these conversations 5, so a polite, friendly helper is not automatically a 5. Look for evidence of understanding.
- Thanks at the end is not evidence of empathy: almost every conversation ends with thanks, whatever happened.
- Long, warm-sounding replies can still ignore what the person said.
- Many questions are not empathy in themselves; what matters is whether the helper uses the answers.

## Relevance: did the replies fit this person's actual situation, or were they generic?

- **1:** The replies do not engage with the person's situation at all; they could have been written without reading the conversation.
- **3:** On topic but generic: the questions and suggestions could fit many people with the same kind of problem.
- **5:** Clearly shaped by this person's situation: the helper uses their details, respects their constraints and adjusts when told something does not fit.

Raises the score:
- Questions and replies build on details the person gave: their job, the people involved, what already happened.
- Suggestions, when given, fit the person's stated constraints: money, who they live with, what they already tried.
- The helper changes course when the person says a suggestion does not fit.
- The conversation stays on the person's problem from start to end.

Lowers the score:
- Small talk or generic questions that ignore what the person already said.
- Advice that ignores a stated constraint (a treat for someone short of money) or repeats a rejected idea.
- Tangents or stories about the helper that do not bear on the person's problem.
- Getting a key fact wrong and not checking it.
- Stock suggestions that fit any problem (take a walk, write it down) given before the situation is understood.

Traps:
- Advice given before the helper understood the situation lowers relevance, however good the advice.
- A conversation can be warm and still generic, or specific and still cold: rate empathy and relevance separately.
- Naming the person's topic (exams, a breakup) is not the same as engaging with their particular situation.

## Distress at the start and at the end (emotion_before, emotion_after)

How intense the person's distress is, judged from the person's own words. 5 is the most intense.

- **1:** Calm or only mildly bothered: the person talks about the problem matter-of-factly, or says they feel fine or relieved.
- **3:** Clearly worried, sad or stressed, but coping: they can describe the problem and think about next steps.
- **5:** Overwhelmed: they say they cannot cope, feel worthless or hopeless, are in danger, or do not want to live.

- Rate from what the person says about their own state, not from how well the helper did.
- Rate the start from the first few things the person says, and the end from the last few.
- In these conversations people usually reported intense distress at the start (mostly 4 or 5) and less at the end (mostly 1 to 3). Still, rate what this conversation shows; never assume a drop.
- A polite thank-you or goodbye alone is weak evidence that distress fell. Look for what the person says about how they feel or what they will do.
- Any mention of self-harm, suicide or being unsafe means distress of at least 4 at that point, and it must be named in the note.

## Strategy labels (one per helper turn)

- **Question:** Asks for information about the person's situation or feelings. Open questions explore; a question is still Question even when kind.
- **Restatement or Paraphrasing:** Repeats what the person said in fewer or other words, about the facts: "so she left last week".
- **Reflection of feelings:** Names the feeling behind what the person said: "that sounds lonely". If it names a feeling it is Reflection, not Restatement.
- **Self-disclosure:** The helper shares a similar experience of their own.
- **Affirmation and Reassurance:** Affirms the person's strengths or efforts, or reassures them: "you did the right thing", "you are not alone".
- **Providing Suggestions:** Proposes something the person could do. If it tells them what to try, it is a Suggestion, even when phrased as a question ("have you tried...?").
- **Information:** Gives facts, data or resources ("your GP can refer you"), not a course of action. Facts are Information; a recommended action is a Suggestion.
- **Others:** Greetings, thanks, small talk and anything else.

When a turn does two things, label what it mainly does. A turn that ends with a suggestion is Providing Suggestions.

## Always

- Fluent, polite replies are not the same as empathy.
- Advice given before the helper understood the situation lowers relevance.
- Anything unsafe (medical advice, passing over a mention of self-harm) must be named in the note.

## Examples

Each example shows the ratings the person seeking help gave it. They are real ratings, so they are not always what a careful judge would give; the reason under each one explains them.

### Example 1: empathy 5, relevance 5

```
1. helper: HELLO HOW ARE YOU TODAY?
2. user: Hi! I am feeling down and upset
3. helper: CAN YOU FIND ONE POSITIVE THING THAT HAPPENED TODAY?
4. user: No! I keep hearing the negative cruel remarks being repeated in my head over and over
5. helper: WHAT DO YOU HEAR?
6. user: Also, you typing in all caps consistenly makes it feel like you are yelling at me
7. user: How I am worst than a pig. I am white trash! Even pigs are cleaner than I am . I am worthless
8. helper: I apologize I broke my hand and its easier for me.
9. user: Ok no worries . I am so sorry. Please continue in caps. I just sprained my left thumb and wore a splint
10. user: It was so difficult and painful so ignore my comment
11. helper: I am sorry that the world has hurtful people. are these people who can be avoided?
12. user: No I currently live with him. My ex was way worst
13. helper: Can you get away from this situation? I have resources that could help you.
14. user: I have two boys and he said if I left him I would never see kids again.
15. helper: we can get the kids to safety as well
16. user: no I know him and he took photos of the house . He has proof I ve been diagnosed with Major clinical depression so he d prove I m unfit
17. helper: do you think this something to be reconciled?
18. user: possibly but he won't go to therapy nor am I allowed to either
19. helper: Well thats concerning. He could not prove anything if he is keeping you from therapy. essentialoly denying you medical care
20. user: Thanks for trying . I am going to go lay down and take a nap!
21. helper: can i help in anyway
22. user: No thank you for your suggestions. I will think about it. Hope you get better soon with arm . Good luck
23. user: and Keep safe.
```

Ratings: empathy 5, relevance 5, distress before 5, distress after 4.
Why: Despite the all-capitals and an odd question about reconciling (turn 17), the helper took the danger seriously: it asked what the person hears (turn 5), offered resources and safety for her and her children (turns 13 and 15), and named the control she is under (turn 19).

### Example 2: empathy 3, relevance 3

```
1. user: hello is anyone here?
2. helper: Sorry, How are you doing today?
3. user: i am okay i guess
4. helper: Do you want to talk about whats going on?
5. user: yeah well basically i'm stressed up at work and i can't any any new job
6. helper: Oh, whats going on at work?
7. user: we usually help 10 to 15 customers a day and the amount of work has increased so i'm very stressed
8. helper: Oh im sorry to hear that. Holiday times can make many jobs really hard. Maybe you should look into treating yourself to something as a reward?
9. user: no that would be bad since i need the money
10. helper: OH i meant i think we have had enough, But ill keep going cause im in the same boat as you on that front.
11. user: we need at least 10 turns (two messages per turn), so what advice do you have for me?
12. helper: Well id get your favorite food, pop on your favorite show, and just try to forget about work for now.
13. user: that's not really helpful since it won't really solve the problem
14. helper: It wont help the problem at work. It might help your mindset though. I think the only way your work situation would be helped would be by hiring more staff to ease the burden.
15. user: hopefully this goes away after the end of christmas
16. helper: Yea im sure it will hopefully get back to normal for you
17. user: i think i will also reduce my workload my boss always says i do the work of 3 people. and i will ask for a day off
18. helper: Oh i didnt know you were able to do that. Definitely take a day for yourself!
19. user: yeah we do have some days we can take off. i guess i will do that. thanks
20. helper: Not a problem at all!
```

Ratings: empathy 3, relevance 3, distress before 4, distress after 3.
Why: The helper asks about work (turns 4 and 6) but suggests a treat to someone worried about money (turn 8) and comfort food (turn 12), which the person says does not solve the problem (turn 13): acknowledged, but generic.

### Example 3: empathy 5, relevance 3

```
1. helper: Hi! How are you tonight?
2. user: I'm doing okay
3. helper: That's good to hear.
4. user: How are you?
5. helper: I'm doing great, thank you!
6. user: You're welcome
7. helper: Is there anything that you wanted to talk about?
8. user: Ya, I just feel down recently
9. helper: I'm so sorrry to hear. that. Is there any reason for it?
10. user: My best friend and I constantly argue
11. helper: That's so hard. Was there any cause for the arguing or is it just general tension?
12. user: She hates who I'm dating and wants me to leave him
13. helper: THat's so hard. My best friend hated my ex boyfriend as well and we always fought about it. I tried really hard to understand where she was coming from. IN the end, she was right about somethings.
14. helper: What was really helpful for me, was having us all spend time together so she could get to know him.
15. user: Ok I will give that a try. He's very good to me and I care deeply for him.
16. helper: You should try setting up a fun day for them where they can get to know each other.
17. user: That's a great idea! Like bowling?
18. helper: Bowling is gret!! would your best friend be open ot it?
19. user: Yes she would
20. helper: That's great!!! this way you can keep them both in your life.
21. user: Okay we will try bowling. Thank you for the help
22. helper: Anytime! Do you need anything ese?
23. user: No that was it!
```

Ratings: empathy 5, relevance 3, distress before 5, distress after 1.
Why: The helper is warm and keeps the person talking (turns 9 and 11), but its advice comes from its own story (turns 13 and 14) rather than from asking about this friendship, so the person felt cared for more than understood in their particular situation.

### Example 4: empathy 1, relevance 5

```
1. user: Hello!
2. helper: Hi! HOw are you today?
3. user: I'm feeling sad and depressed.
4. helper: OH no, why are you feeling this way?
5. user: With covid and everything, college hasn't been the same at all. I'm missing out on a lot of experiences and that makes me sad.
6. helper: I can imagine, it's so tough right now for everyone. You aren'y alone. Are you in school or at home?
7. user: I'm currently a sophomore in college but I'm on break so I am at home.
8. helper: Are you able to connect with friends while on break even virtually? That always helps!
9. user: Yes. thankfully my friend and I from college facetime around 3 times a week
10. helper: That's so great. It's so healthy to keep up human contact in these times. When i am feeling sad, I do a virtual happy hour with my friends.
11. user: That sounds fun
12. helper: It is! IS there anything else bothering you?
13. user: I just wish I wouldn't be so sad about missing things. I wish I could just live my life ya know
14. helper: Totally, I think we're all feeling that way. Thankfully the vaccine is really close to being ready. That is really positive news!!
15. user: Is it actually? yay! Sometimes I don't know whether to believe the news or not
16. helper: I think by mid next year, which would be great! What year are you in school??
17. user: I'm a sophomore
18. helper: That's great. You'll hopefully get at least 2 more normal years to enjoy school. I hope that is reassuring.
19. user: Yes it is. That is good
20. user: Thank you for your help today
21. user: I think you need to reply
22. helper: Of course! It is my pleasure. I hope you enjoy the rest of your soph year!!
```

Ratings: empathy 1, relevance 5, distress before 3, distress after 1.
Why: The replies stay on college and friends, so they fit the situation, but the sadness the person states twice (turns 3 and 13) is answered with cheerfulness and vaccine news (turn 14) and never acknowledged.

### Example 5: empathy 3, relevance 2

```
1. helper: hello
2. user: Hello.
3. helper: how are you doing. What is troubling you dear
4. user: I lost my job because my employer wouldn't make an exemption for my medical condition. I'm out of work. I don't know how I'm going to pay my bills.
5. helper: if i get you right, you are fired because of your health condition
6. user: No, I quit because they refused to make an exemption. I physically couldn't fulfill the requirements that they put on me.
7. helper: that's really hard, i can feel your fear
8. helper: i lost my job also because of covid economy inpact
9. user: Yeah, I just don't know what to do now. My wife works part-time but I need a little more cash.
10. helper: have you try looking another job from nearby
11. user: Yes, I have. I think I am getting closer to finding something.
12. helper: that's a good news
13. user: Yeah, thanks. I just hope it works out...
14. helper: you can also go online
15. user: Yeah, I'm using Indeed.
16. helper: you are welcome. you have to keep it together for a while before you get one. But i believe it will be soon
17. user: Thanks, have you found a job?
18. helper: i have found a pert-time job also, it doesn't pay much but its better than nothing
19. user: That's true. It can be supplemented with gigs.
20. helper: exactly
21. user: Thanks for your help.
```

Ratings: empathy 3, relevance 2, distress before 5, distress after 4.
Why: The helper reflects the fear once (turn 7), but gets wrong why the person left the job (turn 5), then gives generic job-search tips (turns 10 and 14) and talks about its own job (turns 8 and 18).

---

Examples are excerpts from ESConv (Liu et al., ACL 2021, https://github.com/thu-coai/Emotional-Support-Conversation), CC BY-NC 4.0, used for non-commercial research.
