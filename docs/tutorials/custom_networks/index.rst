.. _Custom_networks_tutorials:

Creating Custom Networks Tutorials
=====================================

These tutorials provide an introductory guide to using `AgileRL <https://github.com/AgileRL/AgileRL>`_
to build custom evolvable modules and networks for arbitrarily complex use-cases.

.. raw:: html

    <style>
     /* CSS styles for a single tile with rounded corners, centered title, and always displayed algorithm list */

     /* Style for the container */
     .tiles {
          display: flex;
          justify-content: center;
          margin-top: 48px;
          margin-bottom: 48px;
          width: 100%;
     }

     /* Style for the tile */
     .tile {
          padding: 0px;
          transition: background-color 0.3s ease;
          text-decoration: none;
          width: 300px;
          height: auto;
          overflow: hidden;
          background-color: transparent;
          border-radius: 7px;
          box-shadow: 0 4px 8px rgba(0, 150, 150, 0.5);
          margin-bottom: 0px;
          text-align: center;
     }

     /* Lighter background color on hover */
     .tile:hover {
          background-color: #48b8b8;
          color: white;
     }

     /* Title styles */
     .tile h2 {
          font-size: 18px;
          margin-top: 20px;
          margin-bottom: 20px;
          padding: 4px;
     }

     /* Learn more link styles */
     .tile a {
          display: block;
          margin-bottom: 0px;
          text-decoration: none;
          color: inherit;
          padding: 0px;
     }

     .tile a:hover {
          color: white;
     }

     .thumbnail-image {
          width: 100%;
          height: auto;
          object-fit: cover;
     }

    </style>

    <div class="tiles">
        <a href="../custom_networks/agilerl_rainbow_tutorial.html" class="tile">
            <img src="../../_static/thumbnails/rainbow_performance.png" alt="Rainbow Performance" class="thumbnail-image">
            <h2>Rainbow DQN</h2>
        </a>
    </div>


.. toctree::
   :maxdepth: 2
   :hidden:

   agilerl_rainbow_tutorial
