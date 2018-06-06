This is my capstone project for the Udacity Machine Learning Nanodegree.

In it, I attempt to make a bot that can drift around the ball in rocket league. Please look at proposal.pdf and report.pdf for more details.

The statistics for a few different settings have been included, so running on champions field will work. If you're trying to run this on a field other than champions, you'll need to first collect at least one sample run for that field using the processes/screen_recorder.py script by drifting manually. The preprocessor will automatically generate statistics and normalize the data when it is done.

If looking at the code, start in agent_runner.py and explore from there. The files in processes and explorations may also be interesting.

When running, this project assumes that rocket league is running with the resolution set to 640x480 in borderless mode and is centered on the screen.

I used python 3.6 on anaconda while developing this.
