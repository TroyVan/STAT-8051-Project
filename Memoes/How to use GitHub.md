# Why GitHub?

This GitHub repository is where we will put all of our code, databases, and documentations. We use GitHub because:

- It leaves a record of all changes made to the files in the repo, similar to the edit history for Wikipedia pages. This lets us revisit what we have done to get to where we are and rewind to an earlier version if something breaks.
- The files can be synced to everyone's computers, enabling each person to work on their own and then merge it in with everyone else's work.

# Downloading the repository

You will need to install [GitHub Desktop](https://desktop.github.com/) to work with this repository (using plain old Git is possible but not recommended unless you know what you're doing). Sign in, select File -> Clone repository, and choose this repository to download it to your computer. Now you can open the files locally or open the project in R.

# Making changes and committng

Once you make some changes to the files and/or add new files, you need to "commit" it so that they are recorded. Otherwise the new files stay on your local computer and no one else will see them. From the left panel in GitHub Desktop, choose the files you want to commit, write a summary of what you have done, and press "Commit to master" (or whatever branch you're working on - more on that later). **If a file is not selected ("staged"), its changes will not be recorded.**

**Commit often, even in the middle of doing something, so that you create a change log and can fall back to an earlier version if necessary, and others can see your work.**

# Pushing and pulling

The files on your local computer is a copy of the repository on GitHub, and will not be automatically synced to the cloud - you need to do the syncing yourself. "Pulling" means downloading new commits published on the GitHub repo to your local computer; "pulling" means publishing your commits to the GitHub repo. You do both on the top bar of GitHub Desktop; if "Fetch origin" appears, click it to check if new commits are out, and push and pull as necessary.

**Make sure to check for, and pull, new commits every time you start working!** Otherwise you may be working on files that have been changed in the cloud, creating problems when you push your commits. You should push your commits right after you make them.

# Branches

If you're working on something that make changes to the existing files or may break existing code, it is advisable to "branch off", creating a copy of the current repository, so that you can work without breaking what already works. From the top bar of GitHub Desktop, clock "Current branch", then "New branch", and give your new branch a name. Now you can change things to your heart's content without changing the master branch. When your work is ready to be merged into the master branch, open a pull request; see below.

**Always ensure you're working on the correct branch** by checking the top bar of GitHub Desktop. If you switch branches, your local files will be replaced with the current files from the branch you're switching to.

# Git and R

R will recognize that Git is used in this project, and you can commit, push, and pull in R. **Make sure you're working on the right branch!**

# Issues

Issues is where we discuss ideas, problems and most other things, and keep track of things to do. It is one of two places where discussions can be had similar to forum threads, the other being pull requests. Thus, **everything that needs to be discussed, and all work that needs to be done, should be posted as an issue, even things that are not "issues" per se**. Use labels to signal what the post is about.

Persons can be assigned to an issue to signal that they're working on it. Some issues, like discussion threads, usually aren't assigned to anyone, but they can be if some work needs to be done before discussion continues. Isuues can be closed if the problem is solved, the to-do item is done, or the discussion comes to a conclusion.

You can reference an issue in a commit message, a comment, or another issue by writing #[issue number], e.g. #3; the referenced issue will then contain a reference to your post. Writing "closes #[issue number]" or similar in a commit message will close that issue.

# Pull requests

Pull requests are requests to merge a branch into the master branch. This is where the proposed changes are checked, vetted for bugs, and discussed on before thay are merged into the master branch.

# Projects

Project boards serve as to-do lists and progress trackers, and work similar to Trello (don't worry if you don't know what that is). Here steps and things that need to be done are filed as To-Do, In Progress, or Done. To add a to-do item, write it down as an issue, and add it to the appropriate project board (remember, all work that needs to be done should be posted as issues).
