.. _Custom_networks_tutorials:

Creating Custom Networks Tutorials
=====================================

These tutorials provide an introductory guide to using `AgileRL <https://github.com/AgileRL/AgileRL>`_
to build custom evolvable modules and networks for arbitrarily complex use-cases.

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
    .tile {
        padding: 0px 0px; ; /* Fixed padding */
        transition: background-color 0.3s ease; /* Smooth transition */
        text-decoration: none;
        width: auto; /* Fixed width */
        height: auto; /* Fixed height */
        overflow: hidden; /* Hide overflow content */
        /* display: flex; /* Use flexbox for content alignment */
        flex-direction: column; /* Align content vertically */
        justify-content: center; /* Center content vertically */
        align-items: flex-start;*/
        background-color: transparent; /* Dark grey background */
        border-radius: 7px; /* Rounded corners */
        box-shadow: 0 4px 8px rgba(0, 150, 150, 0.5);
        margin-bottom: 0px;

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
        font-size: 18px; /* Adjust the font size */
        text-align: center; /* Center title text */
        margin-top: 20px;
        margin-bottom: 20px;
        padding: 4px 4px 4px 4px;

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

   </style>


   <div class="tiles article">
     <a href="../custom_networks/agilerl_rainbow_tutorial.html" class="tile">
          <img src="../../_static/thumbnails/rainbow_performance.png" alt="Rainbow Performance" class="thumbnail-image">
          <h2>Rainbow DQN</h2>
     </a>
     <a href="../custom_networks/agilerl_simba_tutorial.html" class="tile">
          <img src="../../_static/thumbnails/simba_thumbnail.png" alt="EvolvableSimBa" class="thumbnail-image">
          <h2>EvolvableSimBa</h2>
     </a>

   </div>

.. toctree::
   :maxdepth: 2
   :hidden:

   agilerl_rainbow_tutorial
   agilerl_simba_tutorial
