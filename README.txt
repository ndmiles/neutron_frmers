Greetings neutron_frmers,

The foundation of gits version control capabilities is its ability to track the history of changes made to files stored in git repositories as a function of time. After the initial creation of the file, any changes made and commited to the online master repository (the neutron_frmers github page) will be recorded and stored in the history of the said file. The information that is stored is the changes that were made (line by line comparison), the date the file was committed and who committed it. This way, if someone updates some portion of their code that is used by others to do specfic things and it causes errors we can pinpoint exactly what has caused the bug and fix it. 


When a git repo is cloned to your computer it will automatically create a copy of the master branch. Online we should have only one branch, the master branch. Once you have copied the master branch to your computer, create a new branch. This will be the branch you edit on. 

Initial Setup:
  -Cloning a repository
    cmd:  git clone https://github.com/ndmiles/neutron_frmers.git  
        -This copies the neturon_frmers.git repo from our page, to a local directory on your computer
         By default, git will name the directory neutron_frmers. To specify a name, just do:
    cmd: git clone https://github.com/ndmiles/neutron_frmers.git dir_name  

  -Creating a new branch
    cmd: git checkout -b branch_name
        -The flag -b, means new branch and so git will create a new branch called 'branch_name' then switch from your current branch to the branch named 'branch_name'
  
  -Linking the new branch to the online repo
    cmd: git push --set-upstream origin test


Basic Commands:
  -Check the status of the current repo you are in, this will show you what files you have modified
    cmd: git status  

  -See what you changed in the files listed in the output of git status
    cmd: git diff

  -If you are content with the changes, add them to be tracked
    cmd: git add filename

  -Once all of the files have been added, commit them
    cmd: git commit -m 'some short/illuminating message about what you have changed'<---if you only made a couple minor ones
    cmd: git commit -F filename <--- If you made a lot of changes, write them in a file and give that as the commit message     
