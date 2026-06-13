---
Created on: 2026-01-28 17:06
Last Modified: 2026-01-28 17:06
File Folder: Programming
tags: 
Parent:
---
# **my coding philosophy**

I have a idea of a desire to get a task done. I make a prototype/proof of concept using AI to see if the concept is viable. Let's say I need a tool to do something. I write very quick and dirty code to push features as fast as possible. Once I stress test and prove that the tool works, saves me time, is actually useful, has a reason to exist, etc... The code written by the AI in this stage is likely to contain a lot of [[AI coded slop]]

Then I can think about making a nice UI for example. 

> [!note]+ The importance of making UI
> Making a UI as a human interactable wrapper for my code is one of the most important aspects of my coding style. Since [[I cannot read code]], and I operate on the notion that users of my tools will by laymen that also cannot read code... aka artists etc. Then making a good UI is one of the most important things to do. One of the best examples of why UI is important is [yt-dlp](https://github.com/yt-dlp/yt-dlp), which I use all the time. This tool is a nightmare to use because most of the features are inaccessible to the end user. As a "lay person", I cannot be expected to read or write in the command line. I even have a [obsidian note](obsidian://open?vault=Obsidian&file=1_MainNotes%2FEvery%20Software%20I%20installed%20to%20windows%2FEvery%20Software%20I%20installed%20to%20windows) to track all my commonly used command lines. It's imperative that my tools contain a easy to use UI because that is the user's I/O. 
> 
> My thought process on making a UI is, "Can I use this tool in 10 months from now?". Lets say I don't need this tool for a extended period of time and then suddenly need to use it. Can I just pick it up and use it? Is there friction there?

I make a UI skeleton, just the UI with no logic connected. This ensures that whatever is being written has a very low amount of slop. After I'm finished with the design and appearance of the UI, then I can implement the features one at a time. This prevents the AI from going crazy since there's a clean scaffolding to hold onto.



> [!think]
> I used to try and refactor my prototype code and add a UI onto that code base. This method only works if you are confident that the code has minimal slop. or else it will backfire. My current method of going as fast as possible thru the prototyping phase allows me to iterate many tools quickly without having to consider the nonsense the AI outputs. we must acknowledge that it's [impossible to write clean code](https://youtu.be/8ncQrGuunHY?t=866). We are always in the prototyping phase to some extent. 





## Clearly Defining your software's scope
Define your scope, design constraints, 
what is this called?

> [!think]
> this thought is incomplete

I was working on [Twinkle Tray](https://github.com/WolfExplode/twinkle-tray) and I was considering a solution to a issue, but decided not to implement the solution because:
"per-pixel processing for highlights compression via GPU (DWM-style) is expected to have a bit of lag/overhead that goes beyond what a lightweight electron app such as twinkle tray should be doing."

It's really common to identify a issue, then tell the AI to fix it. but we must remember to consider the scope of our project. our software needs to "stay in its lane" and not overstep its computational footprint. 
depending on the app, each app we make has a "processing budget". For a lightweight background app, the processing budget is miniscule. We should not be implementing features that bog down the user's system. this is not acceptable software design. 

Before we say "yes AI, fix my issue" we must first ask and read how the AI will fix the issue. Then determine if that fix is acceptable. Not all problems should be fixed.





## Overly defensive programing
I realize that the AI often puts overly defensive programing to try and catch errors before they occur. I've realized that it's better to let the bugs and edge cases occur because they tell the user that there's a fundamental issue with their logic. If we use overly defensive programing it can mask poor fundamental logic. If we let the bugs occur naturally, the programmer can realize that something is wrong and go investigate. "If something should fail it should fail loudly". you never want silent or unnoticed bugs in the codebase because that causes the programmer's mental image of the codebase to drift from what is actually happening in the code.  

Basically, overly defensive programing can cause everything to appear like it's implemented correctly while in reality the AI just put so many band-aid catches so it looks functional even though the root cause is improper logic. 
I'm hesitant to call "improper logic" a bug because it's a symptom of a unfinished idea. or if not enough thinking has been done before implementation. or if the AI implemented the idea in a very poor way. It doesn't necessarily indicate that the initial idea is a bad idea, just we haven't thought enough about it yet to make sure it gets implemented correctly. 

My mental model gets anchored to my prompt, not the output. Since I cannot read code, I work in prompt space. I just assume that if the AI misunderstands me it wouldn't work. I never thought to check "the AI implemented my idea in a different way then I intended and the defensive code it wrote prevented me from realizing"

so tldr, it's not good to have overly defensive programing because it masks poorly implemented code. 

When reviewing AI-generated code, look for the ratio of "defensive" lines to "operational" lines. If a 20-line function has 12 lines of null checks, type guards, and try/catches, that's a smell. Not because the guards are wrong individually, but because their density suggests the AI didn't understand the data flow enough to know _which_ specific invariants matter.




## When it comes to vibe coding, the first bug is always intent
Is the code doing what you intended it to do, the way you intended it?
many times, when you ask the AI to implement your idea, it will do it. and when you test it out. it will kinda work. so you don't think too much about it. Then a bug occurs and it's confusing because if the AI implemented what you intended this bug should not occur. then you look at the code and discover the slop. 

When something goes wrong, we go looking for the bug. but what is the bug? I argue, the first bug is always unverified intent. 

Most of the time, this is caused because you did not hand hold the AI. or you did not fully explain the implementation. or you did not think thru the logic.  
In your brain, there's a path from A to B. if the AI takes a different path, it doesn't matter if it technically works. or if the end user can't see a difference because the end result is the same, I still consider any deviation from the intended path to be a bug. 

We can't really fully explain our reasoning for making a specific decision. from a outside observer, the decision to do x might look like nonsense or seem redundant. but as the developer, we know what this code is being used for, what customer it has to serve, the design requirements it must meet, the optimization targets, the scope, etc. As the developer, I know where I want to take this codebase. where the architecture must go. 
I understand that it's kinda impossible to explain to the AI all of these thoughts and considerations. After all, all the AI has to do is to implement X and not question it. It doesn't need to know the full set of details, it even works better if you don't flood it with unnecessary info. 

It must also be said, that often times, the AI will come up with a solution that's better than the one you intended. in that case, to fix the "bug of mismatched intent", we must have the AI explain the code to us to update our own mental model. To align our intent with the AI's instead of the other way around. Sometimes, the AI's deviation can identify a gap in your own thinking. Something you haven't considered yet etc. 

If mental model alignment isn't done, it will be a form of technical debt. In a non traditional sense, but I consider it technical debt. Intent-checking should be done proactively, right after generation. but most people only do it reactively, when something breaks. It feels like a chore, but we must do our due diligence. 

It also must be said, that vibe coding encourages not having a clear intent in the first place. You prototype into existence, and the AI's implementation becomes your intent retroactively. For exploration and prototyping this isn't necessarily a bad thing though. If never actually decided what you wanted, then that's a you issue and not the AI's fault. 










## I don't want to read code
I want to live the rest of my life not being able to read code. does that sound crazy? AI will do that shit anyways. I want to be doing [real work](https://youtu.be/TjfWEajoESc?t=29) 


## Which AI service provider to use?
I don't like using "stronger AI" for everything. I think nowadays, most AI should be able to write the code you want it to write if you prompt it properly. I've noticed that "Auto" on cursor can do just about everything I need it to do.
I think if the AI doesn't know what you mean, it's a indication that you need to work on:
- Your communication skills
- Your knowledge of the issue
- Clarify your design requirements
- you're just saying nonsense, aka you don't know wtf you're talking about. or maybe you are saying x is a problem when it really isn't. like if you say there isn't x in the codebase without checking to make sure first, then the AI will burn reasoning tokens to realize that you just lied.

It's good to stop thinking in terms of "this is a limitation of AI" and instead think "this is a natural limitation that arises from the process itself"



## Debugging

> [!say]
> Add some console logging and run it here. then I'll load a image into the UI and process. that will show you what the console logs and you will be able to determine the issue

> [!say]
> Add logging to show a full stack trace for relevant part so we can see what's happening


## AI Prompts
> [!note to AI]
> - Do not over-engineer a solution, keep it simple
> - Do not remove debugging code unless specified by the user
> - Try to avoid further abstracting my code unless specified by the user
> - Try to avoid further segmenting my code unless specified by the user
> - After you implement something, remove all redundant and not needed code to keep the codebase clean
> - Don't maintain backwards compatibility when refactoring. 
> - If you are confused or have any questions, ask the user before you implement something
> - Avoid adding too much defensive programming, if a bug or error occurs as part of the implementation of a new idea, just let it happen. the user will let you know.
> - Try to speak plain layman's English. the user does not read the codebase. Use the technical terms the user speaks.
> - When you implement a decision made by the user, make sure to document/docstring the reason why the decision was made. Make sure to document why decisions are made and not how the code works.

> [!say]
> now let's play a game of code golf. try to remove redundant code and minimize the lines of code in our codebase. If something can be simplified, do it. If the architecture of our code seems too complex for what it's trying to do, tell me. places where we put too much defensive coding that's likely not needed. etc.


> [!say]
> If something isn't in the docs, say so
> Before responding to any Blender-related request, always read the relevant files in /your/docs/folder/ first. 
> Do not rely on training memory for node names, socket types, or parameter values

> [!say]
> make a list of possible issues. then we will try each fix and 
> to avoid making the code messy with unwanted fixes, we will revert and try the next fix until we find what the issue is.

Often times, we run into a issue but are unsure of how to solve it. We make the AI implement 10 different solves and eventually we solve the issue. then we commit all changes but in reality, only the last "solution" was the correct one. all other "solutions" to the issue were dead ends etc. if we do not clean up after ourselves and remove unneeded code, our codebase will become bloated with crap. It's important to recognize which of these solutions actually solved the issue and commit nothing else. 

The AI loves to do [[defensive programing]] and bloat our our project. Just the other day, I was starting on a new project and the codebase was in its infancy. just starting out. just starting to implement features etc.
the AI wrote "backwards compatibility" code into our codebase of 400lines. wtf? literally slop. 





> [!say]
> Let's do a codebase audit. Can you find areas of improvement to my codebase to make it more maintainable?
> 

> [!say]
> that was a large refactor, let's double check if all the decisions we made make sense in the larger context of what we are trying to do.

> [!say]
> There are bits of logic that I implemented to the script that likely do not need to be there. I come up with a idea to solve a problem, then later I come up with a better idea but left the initial solution in the code. Therefore there may be redundancy in logic that's still being used in the call stack but from a practical standpoint the logic is redundant. 
> 
> Can you scan our codebase for such cases if they do exist

> [!say]
> Sometimes, a module name doesn't match its actual contents. Can you scan our codebase for such cases if they do exist

> [!say]
> can you scan the codebase for lingering circular dependancies etc.
> observe the current structure and detemine if it needs to be done the way it is. should we further refactor for a more maintanable structure etc...

> [!say]
> can you scan the codebase for dead/redundant code? any stale or outdated comments?

> [!say]
> there may be a lot of redundant code in our codebase. can you scan our code to check for areas of improvements?  
> places where we put too much defensive coding that's likely not needed. etc.
> and list anything you are not sure of so you can research how blender behaves in the future
> 
> ok do some research on what you are unsure about and tell me what you find. after that, we can begin to tim down our codebase
> 
> ok trim down our codebase

> [!say]
> We should be writing code/text that makes it easy for AI to decipher and replace. Previously, we had difficulty with, and replacing/editing code using AI. Replace all high-risk characters with their ASCII equivalents so the codebase would be all-ASCII in its source text (string literals + comments). double check areas of concern. 

> [!say]
> I'm using the graphical display to debug and understand how my code works. I don't want anything for display only. display should always show what data the algorithm uses/processes. can you double check that all display data is direct data directly used by the algorithm? I don't want anything to display as one thing but then never get used in the algorithm's logic. that's nonsensical. leading to many confusing bugs

> [!say]
> seems like you're taking a while to think, anything I could help you figure out?
> if you are confused about something ask me questions and I'll try my best to answer.





yeah, so imagine you don't speak Chinese 
and you're in china
taking to a guy who's talking to you in chinese
but  you don't know wtf he's saying
but from the way he's saying it, you can kinda follow along
and kinda just vibe it out
so if you pay attention you can fully understand him
and in this case, he can understand english but can only reply in chinese
so it's almost a one sided conversation, info only flows from your mouth to his ears
but since you don't speak chinese you can't really keep the conversation going
so it's like you're speaking to yourself
but like I said, if you really pay attention you can kinda fully understand the vibe
that's agentic programing in a nutshell
so you ask probing questions like "how does this codebase handle the multires modifier since blender's API doesn't allow us to probe it?"
and then the AI will say some shit like "blah blah possible to do xxx blah blah but slow blah blah you already do blah blah do you want me to change your implementation?"

and then I will be like, oh shit, we already do blah blah, wow. I didn't consider that I already had the infrastructure to implement my idea cleanly. then i will update my prompt to say "let's do x" and the ai will be able to implement it cleanly. cuz it spoke Chinese to me and I understood the vibe. therefore I know it's a good idea to do x.

and so if you don't do this, and you loose the plot.
it's like, "oh just implement y" because In my brain, I'm thinking I already have the infrastructure in my codebase to support this implementation. but in reality, I didn't keep up with what the AI has implemented so far so it will fail
if I keep read the AI's entire thought process, the little grey text, the thinking tokens. I will be able to understand and vibe it out. like wtf why is the AI spending so much time thinking? something's wrong!
then I can speak to it, and it will speak chinese to me
and then I will vibe it out
and then realize ohhh. I'm dumb, I failed to consider xyz

When I ask the AI to implement something, I always make sure I have a idea of how many lines of code the implementation should be
And which files/ where should be impacted
If I don't know these things, I make sure to ask the AI more questions such as "how could this be implemented"
And the it will speak Chinese to me
And I vibe the meaning out
the phrase "how could this be implemented" is a probing phrase.

it exposes three things:
how large the implementation should be,
which files should be impacted, 

but most importantly, 
and updates your understanding of the issue at hand.
so if the AI starts saying some shit that just feels off
it's not a indication that the AI is dumb
it's a indication that you are dumb
it means you don't yet fully understand the implications of your query and need to reevaluate your approach
the AI is a no feeling no thinking machine. you tell it to do something dumb, it will do it
so if it does something dumb, 90% of the time, it's because you are dumb
don't blame the AI for something you told it to do














## Runtime regressions
A software regression is a type of software bug where a feature that has worked before stops working correctly





