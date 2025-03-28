.. _LLM_Finetuning_tutorials:

LLM Finetuning Tutorials
========================

These tutorials provide an introductory guide to using `AgileRL <https://github.com/AgileRL/AgileRL>`_
for finetuning LLMs.


.. raw:: html

   <style>
    /* CSS styles for tiles with rounded corners, centered titles, and always displayed algorithm list */

    /* Style for the container */
    .tiles {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 0.25fr));
        grid-auto-rows: 200px; /* 2 rows */
        gap: 25px; /* Adjust the gap between tiles */
        margin-top: 48px;
        margin-bottom: 48px;
        width: 100%;
        align-content: start;
        /*height: auto;
    }

    /* Style for each tile */


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




    /* Learn more link styles */
    .tile a {
        display: block;
        margin-bottom: 0px; /* Adjust the margin */
        text-decoration: none;
        /*color: white; /* Link color */
        text-align: center; /* Center link text */
        padding: 0px 0px;
    }

    .tile a:hover {
        color: white; /* Link color on hover */
    }

    .thumbnail-image {
        width: 100%;
        height: 60%;
        object-fit: cover;

    }
    /* Style for each tile */
    .tile {
        padding: 4px 4px;
        transition: background-color 0.3s ease;
        text-decoration: none;
        width: auto;
        height: auto;
        overflow: hidden;
        display: flex; /* Enable flexbox */
        flex-direction: column;
        justify-content: center; /* Center vertically */
        align-items: center; /* Center horizontally */
        background-color: transparent;
        border-radius: 7px;
        box-shadow: 0 4px 8px rgba(0, 150, 150, 0.5);
        margin-bottom: 0px;
    }

    /* Title styles */
    .tile h2 {
        font-size: 18px;
        text-align: center;
        margin: 0; /* Remove margins */
        padding: 4px;
        width: 100%; /* Ensure the h2 takes full width */
    }

   </style>


   <div class="tiles article">
      <a href="../llm_finetuning/grpo_finetuning.html" class="tile">
         <h2>GRPO - Finetuning</h2>
      </a>
      <a href="../llm_finetuning/grpo_hpo.html" class="tile online">
         <h2>GRPO - Finetuning with HPO</h2>
      </a>
   </div>


.. toctree::
   :maxdepth: 2
   :hidden:

   grpo_finetuning
   grpo_hpo
