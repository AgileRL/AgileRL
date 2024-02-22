Get started
===========

Explore our Algorithms!
-----------------------

.. raw:: html

   <style>
    /* CSS styles for tiles with rounded corners, centered titles, and always displayed algorithm list */

    /* Style for the container */

   @media (max-width: 750px) {
      .tiles_2 {
         display: grid;
         grid-template-columns: 100%; /* 2 columns */
         grid-auto-rows: 0% 50% 50% 0%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 0px; /*48px;*/
         margin-bottom: 58px;
         width: 100%;
         align-content: center;
         height: auto;
         min-height: 185px;
      }

      .tiles_3 {
         display: grid;
         grid-template-columns: 100%; /* 3 columns */
         grid-auto-rows: 33%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 48px;
         margin-bottom: 72px;
         width: 100%;
         align-content: start;
         height: auto;
         min-height: 185px;
      }
   }

   @media (min-width: 750px) {
      .tiles_2 {
         display: grid;
         grid-template-columns: 16.5% 33% 33% 16.5%; /* 2 columns */
         grid-auto-rows: 100%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 0px; /*48px;*/
         margin-bottom: 58px;
         width: 100%;
         align-content: center;
         height: auto;
         min-height: 185px;
      }
      .tiles_3 {
         display: grid;
         grid-template-columns: 33% 33% 33%; /* 3 columns */
         grid-auto-rows: 100%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 48px;
         margin-bottom: 25px;/*58px;*/
         width: 100%;
         align-content: start;
         height: auto;
         min-height: 185px;
      }
   }

    /* Style for each tile */
    .tile {
        padding: 0px 20px 20px; ; /* Fixed padding */
        transition: background-color 0.3s ease; /* Smooth transition */
        text-decoration: none;
        width: auto; /* Fixed width */
        height: auto; /* Fixed height */
        overflow: hidden; /* Hide overflow content */
        display: flex; /* Use flexbox for content alignment */
        flex-direction: column; /* Align content vertically */
        /*justify-content: center; /* Center content vertically */
        /*align-items: flex-start;*/
        background-color: transparent; /* Dark grey background */
        border-radius: 7px; /* Rounded corners */
        position: relative; /* Relative positioning for algorithm list */
        box-shadow: 0 4px 8px rgba(0, 150, 150, 0.5);
    }

    .column {
    flex: 1; /* Equal flex distribution */
    width: 50%; /* 50% width for each column */
    display: flex;
    flex-direction: column;
    /* Additional styles */
   }

    /* Lighter background color on hover */
    .tile:hover {
        background-color: #48b8b8; /* Lighter grey on hover */
        color: white;
    }

    /* Title styles */
    .tile h2 {
        margin-bottom: 8px; /* Adjust the margin */
        font-size: 24px; /* Adjust the font size */
        text-align: center; /* Center title text */
    }

   .tile p {
         margin-top: 12px;
         margin-bottom: 8px; /* Adjust the margin */
         font-size: 16px; /* Adjust the font size */
         text-align: left;
         word-wrap: break-word;
      }


    /* Learn more link styles */
    .tile a {
        display: block;
        margin-top: 8px; /* Adjust the margin */
        text-decoration: none;
        /*color: white; /* Link color */
        font-size: 14px; /* Adjust the font size */
        text-align: center; /* Center link text */
    }

    .tile a:hover {
        color: white; /* Link color on hover */
    }
   </style>

   <div class="tiles_3 article">
      <a href="../on_policy/index.html" class="tile on-policy article">
         <h2>On-policy</h2>
         <p>
               Algorithms: PPO
         </p>
      </a>
      <a href="../off_policy/index.html" class="tile off-policy">
         <h2> Off-policy</h2>
            <p>
                  Algorithms: DQN, Rainbow DQN, TD3, DDPG
                  <!-- Add more algorithms as needed -->
            </p>
      </a>
      <a href="../offline_training/index.html" class="tile online">
         <h2>Offline</h2>
         <p>
               Algorithms: CQL, ILQL
               <!-- Add more algorithms as needed -->
         </p>
      </a>
   </div>
   <div class="tiles_2 article">
      <div></div>
      <a href="../multi_agent_training/index.html" class="tile multi-agent">
         <h2>Multi Agent</h2>
         <p>
               Algorithms: MADDPG, MATD3
               <!-- Add more algorithms as needed -->
         </p>
      </a>
      <a href="../bandits/index.html" class="tile bandit">
         <h2>Contextual Bandits</h2>
         <p>
               Algorithms: NeuralUCB, NeuralTS
               <!-- Add more algorithms as needed -->
         </p>
      </a>
   </div>


.. _install:

Install AgileRL
---------------

To use AgileRL, first download the source code and install requirements.

Install as a package with pip:

.. code-block:: bash

   pip install agilerl

Or install in development mode:

.. code-block:: bash

   git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
   pip install -e .
