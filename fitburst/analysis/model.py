"""
Object for Computing and Updating Models of Dynamic Spectra

The SpectrumModeler() object is designed to compute models of dynamic
spectra based on parameter values, and handle the updating of one or more
model parameters. The updating/retrieval methods are used in the fitburst
fitter object, and are written to handle user-specified fixing of parameters.
"""

import sys

import numpy as np

import fitburst.routines as rt
from fitburst.backend import general
from concurrent.futures import ThreadPoolExecutor


class ModelSlice:
    def __init__(
        self,
        freqs: float,
        times: float,
        comp_start: int,
        comp_end: int,
        id: int,
        dm_incoherent: float = 0.0,
        factor_freq_upsample: int = 1,
        factor_time_upsample: int = 1,
        is_dedispersed: bool = False,
        is_folded: bool = False,
        scintillation: bool = False,
        save_intermediate: bool = False,
    ):
        self.dm_incoherent = dm_incoherent
        self.factor_freq_upsample = factor_freq_upsample
        self.factor_time_upsample = factor_time_upsample
        self.freqs = freqs
        self.is_dedispersed = is_dedispersed
        self.is_folded = is_folded
        self.scintillation = scintillation
        self.times = times
        self.save = save_intermediate

        self.start_idx = comp_start
        self.end_idx = comp_end
        self.num_components = comp_end - comp_start
        self.worker_id = id

        # derive some additional data needed for downstream fitting.
        self.num_freq = len(self.freqs)
        self.num_time = self.times.shape[-1]
        self.res_freq = np.fabs(self.freqs[1] - self.freqs[0])
        self.res_time = np.fabs(self.times[..., 1] - self.times[..., 0])

        if self.times.ndim == 1:
            self.has_freq_dependent_times = False

        # define all *fittable* model parameters first.
        # NOTE: 'ref_freq' is not listed here as it's a parameter that is always held fixed.
        self.parameters = [
            "amplitude",
            "arrival_time",
            "burst_width",
            "dm",
            "dm_index",
            "scattering_timescale",
            "scattering_index",
            "spectral_index",
            "spectral_running",
        ]

        # now instantiate parameter attributes and set initially to NoneType.
        for current_parameter in self.parameters:
            setattr(self, current_parameter, None)

        if self.save:
            # now instantiate the structures for per-component models, time differences, and temporal profiles.
            # (the following are used for computing derivatives and/or per-channel amplitudes.)
            self.amplitude_per_component = np.zeros(
                (self.num_freq, self.num_time, self.num_components), dtype=float
            )

            self.spectrum_per_component = np.zeros(
                (self.num_freq, self.num_time, self.num_components), dtype=float
            )

            self.timediff_per_component = np.zeros(
                (self.num_freq, self.num_time, self.num_components), dtype=float
            )

            self.timeprof_per_component = np.zeros(
                (self.num_freq, self.num_time, self.num_components), dtype=float
            )

    def compute_model(self, data: float | None = None) -> float:
        """
        Computes the model dynamic spectrum based on model parameters (set as class
        attributes) and input values of times and frequencies.

        Parameters
        ----------
        data : float, optional
            an array of time_averaged data values to be used for per-channel normalization
        """

        # pylint: disable=no-member,too-many-locals

        # if scintillation modeling is desired but data aren't provide, then exit.
        if self.scintillation and data is None:
            sys.exit("ERROR: scintillation modelling is desired by data are missing!")

        # create an upsampled version of the current frequency label.
        # even if no upsampling is desired, this will return an array
        # of length 1.
        upsampled_frequencies = rt.manipulate.upsample_1d(
            self.freqs, self.res_freq, self.factor_freq_upsample
        )  # shape: (freqs * upsampling,)

        # first, compute "full" delays for all upsampled frequency labels.
        dm_delay = rt.ism.compute_time_dm_delay(
            self.dm_incoherent + self.dm[0],
            general["constants"]["dispersion"],
            self.dm_index[0],
            upsampled_frequencies,
            self.ref_freq[0],
        )  # shape: (freqs * upsampling,)

        # then compute "relative" delays with respect to central frequency.
        if not self.has_freq_dependent_times:
            # if the time axis is shared across all freqs,
            # incoherent dedispersion has already aligned to
            # `ref_freq`. Remove that alignment here so the
            # residual DM term is applied consistently per channel.
            dm_delay -= np.repeat(
                rt.ism.compute_time_dm_delay(
                    self.dm_incoherent,
                    general["constants"]["dispersion"],
                    self.dm_index[0],
                    self.freqs,
                    self.ref_freq[0],
                ),  # shape: (freqs,)
                self.factor_freq_upsample,
            )  # shape: (freqs * upsampling,)

        # create a 2D array for each component containing the times and remove the toa.
        toas = np.asarray(self.arrival_time, dtype=float)  # shape: (comps, )

        if self.has_freq_dependent_times:
            times_minus_toa = self.times[None, ..., :] - toas[:, None, None]
            ncomp, nfreq, ntime = times_minus_toa.shape

            # Transpose the time axis up front, before combining axis 0 and 1
            flat_T = (
                times_minus_toa.transpose(2, 0, 1).reshape(ntime, ncomp * nfreq).copy()
            )

            # Upsample the data
            flat_up_T = rt.manipulate.upsample_1d(
                flat_T, self.res_time, self.factor_time_upsample
            )

            # Transform back into tile
            tile = flat_up_T.reshape(-1, ncomp, nfreq).transpose(
                1, 2, 0
            )  # shape: (comps, freqs * upsampling, time * upsampling)

            # And since we upsampled, we need to share the duplicated axis:
            tile = np.repeat(tile, self.factor_freq_upsample, axis=1)
        else:
            times_minus_toa_T = self.times[:, None, None] - toas[None, :, None]
            upsampled_times_T = rt.manipulate.upsample_1d(
                times_minus_toa_T, self.res_time, self.factor_time_upsample
            )  # shape: (time * upsampling, comps, freqs)

            upsampled_times = upsampled_times_T.transpose(
                1, 2, 0
            )  # shape: (comps, freqs, time * upsampling)

            tile = np.repeat(
                upsampled_times,
                upsampled_frequencies.shape,
                axis=1,
            )

        # remove the (relative) DM delay; dm_delay is (nfreq_up,), so this broadcasts for both cases.
        tile = tile - dm_delay[None, :, None]

        # Save the time difference for each component:
        if self.save:
            self.timediff_per_component = rt.manipulate.downsample_tile(
                tile, [1, self.factor_freq_upsample, self.factor_time_upsample]
            )  # shape: (comp, freqs, time)

        # next, compute and store raw temporal profile.
        profile = self.compute_profile(
            tile,
            0,  # we already remove the ToAs a few lines ago. No need to do it again.
            self.scattering_timescale[0],
            self.scattering_index[0],
            self.burst_width,
            upsampled_frequencies,
            self.ref_freq[0],
            self.is_folded,
        )

        # Again, save the profile for each component:
        if self.save or self.scintillation:
            self.timeprof_per_component = rt.manipulate.downsample_tile(
                profile, [1, self.factor_freq_upsample, self.factor_time_upsample]
            )

        # Compute the spectrum and add it to the profile:
        profile *= rt.spectrum.compute_spectrum_rpl(
            upsampled_frequencies,
            self.ref_freq[0],
            self.spectral_index,
            self.spectral_running,
        )[:, :, None]

        # Downsample the profile for a final time:
        profile = rt.manipulate.downsample_tile(
            profile, [1, self.factor_freq_upsample, self.factor_time_upsample]
        )

        if self.save:
            # Save the amplitudes scaled up with the SED.
            self.amplitude_per_component = (
                rt.spectrum.compute_spectrum_rpl(
                    self.freqs,
                    self.ref_freq[0],
                    self.spectral_index,
                    self.spectral_running,
                )
                * (10 ** np.array(self.amplitude))[:, None]
            )  # shape: (comp, freqs)

        self.spectrum_per_component = (
            profile * (10 ** np.array(self.amplitude))[:, None, None]
        )

        # if scintillation is enabled compute per channel amplitude:
        if self.scintillation:
            amps_2d = rt.ism.compute_amplitude_per_channel(
                data, self.timeprof_per_component
            )
            current_amplitudes = amps_2d[:, :, np.newaxis]

            if self.save:
                self.amplitude_per_component = current_amplitudes

            self.spectrum_per_component = (
                current_amplitudes * self.timeprof_per_component
            )

        return self.spectrum_per_component

    def compute_profile(
        self,
        times: float,
        arrival_time: float,
        sc_time_ref: float,
        sc_index: float,
        width: float,
        freqs: float,
        ref_freq: float,
        is_folded: bool = False,
    ) -> float:
        """
        Returns the temporal profile, depending on input values of width
        and scattering timescale.

        Parameters
        ----------
        times : float
            One or more values corresponding to time

        arrival_time : float
            The arrival time of the burst

        sc_time_ref : float
            The scattering timescale of the burst (which depends on frequency label)

        sc_index : float
            The index of frequency dependence on the scattering timescale

        width : float
            The intrinsic temporal width of the burst

        freqs : float
            The index of frequency dependence on the scattering timescale

        ref_freq : float
            The index of frequency dependence on the scattering timescale

        is_folded : bool, optional
            If true, then the temporal profile is computed over two realizations and then
            averaged down to one (in order to allow for wrapping of a folded pulse shape)

        Returns
        -------
        profile : float
            One or more values of the temporal profile, evaluated at the input timestamps
        """

        # pylint: disable=too-many-arguments,no-self-use

        # if data are "folded" (i.e., data from pulsar timing observations),
        # model at twice the timespan and wrap/average the two realizations.
        # this step accounts for potential wrapping of pulse shape.
        times_copy = times.copy()

        if is_folded:
            # Use the last axis for time extension
            res_time = np.diff(times_copy, axis=2)[:, :, 0]
            start = times[:, :, -1] + res_time
            stop = times[:, :, -1] + (res_time * times.shape[2])

            times_extended = np.linspace(
                start=start, stop=stop, num=times.shape[2], axis=2
            )
            times_copy = np.append(times, times_extended, axis=2)

        # compute either Gaussian or pulse-broadening function, depending on inputs.
        profile = np.zeros(times_copy.shape, dtype=float)
        sc_time = sc_time_ref * (freqs / ref_freq) ** sc_index
        normalize = general["flags"]["normalize_pbf"]

        if np.any(sc_time > 0.0):
            # Setup the masking
            threshold = (np.array(width) * -5)[:, None, None]  # shape: (comp, 1, 1)
            mask = times_copy < threshold

            replacement_value = np.tile(
                threshold,
                (1, times.shape[1], times.shape[2]),
            )  # shape: (comp, freqs, time)

            times_copy[mask] = replacement_value[mask]

            profile = rt.profile.compute_profile_pbf(
                times_copy,
                arrival_time,
                width,
                freqs,
                ref_freq,
                sc_time_ref,
                sc_index=sc_index,
                normalize=normalize,
            )
        else:
            profile = rt.profile.compute_profile_gaussian(
                times_copy, arrival_time, width
            )

        # if data are folded and time/profile data contain two realizations, then
        # average along the appropriate axis to obtain a single realization.
        if is_folded:
            profile = profile.reshape(
                times.shape[0], times.shape[1], 2, times.shape[2]
            ).mean(axis=2)

        return profile

    def update_parameters(
        self,
        model_parameters: dict,
        global_parameters: list | None = None,
    ) -> None:
        """
        Overloads parameter values stored in object with those supplied by the user.

        Parameters
        ----------
        model_parameters : dict
            a Python dictionary with parameter names listed as keys, parameter values
            supplied as lists tied to keys.

        global_parameters : list = ["dm", "scattering_timescale"]
            a Python list with the parameter names that are kept constant across components.

        Returns
        -------
        None : None
            this method overloads class attributes.
        """

        # first, overload attributes with values for supplied parameters.
        if global_parameters is None:
            global_parameters = ["dm", "scattering_timescale"]
        for current_parameter, current_value in model_parameters.items():
            setattr(self, current_parameter, current_value)

            # if number of components is 2 or greater, update num_components attribute.
            if len(current_value) > 1:
                num_components = len(current_value)

                if current_parameter in global_parameters:
                    setattr(
                        self,
                        current_parameter,
                        [current_value[0]] * num_components,
                    )


class SpectrumModeler:
    """
    A Python structure that contains all information regarding parameters that
    describe and are used to compute models of dynamic spectra.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        freqs: float,
        times: float,
        dm_incoherent: float = 0.0,
        factor_freq_upsample: int = 1,
        factor_time_upsample: int = 1,
        num_components: int = 1,
        is_dedispersed: bool = False,
        is_folded: bool = False,
        scintillation: bool = False,
        verbose: bool = False,
        save_intermediate: bool = False,
        num_threads: int = 1,
    ) -> None:
        """
        Instantiates the model object and sets relevant parameters, depending on
        desired model for spectral energy distribution.

        Parameters
        ----------

        freqs: float
            The frequency channels in the spectrum (including masked ones)

        times: float
            The time samples in the spectrum

        dm_incoherent : float, optional
            The DM used to incoherently dedisperse input data; this is only used if the
            'is_dedispersed' argument is set to True

        factor_freq_upsample : int, optional
            The factor to upsample each frequency label into an array of subbands

        factor_time_upsample : int, optional
            The factor to upsample the array of timestamps

        is_dedispersed : bool, optional
            If true, then assume that the dispersion measure is an 'offset' parameter
            and computes the relative dispersion for non-zero offset values

        is_folded : bool, optional
            If true, then the temporal profile is computed over two realizations and then
            averaged down to one (in order to allow for wrapping of a folded pulse shape)

        num_components : int, optional
            The number of distinct burst components in the model

        scintillation : bool, optional
            if true, the compute per-channel amplitudes using input data.

        verbose : bool, optional
            If true, then print parameter values during each function call
            (This is mainly useful to gauge least-squares fitting algorithms.)

        save_intermediate : bool, optional
            If true, this will save intermediate arrays for debugging.
        num_threads : int, optional
            Sets the number of threads used for the parallelized computations of the model.
        """

        # pylint: disable=too-many-arguments,too-many-locals

        # first define model-configuration parameters that are not fittable.
        self.dm_incoherent = dm_incoherent
        self.factor_freq_upsample = factor_freq_upsample
        self.factor_time_upsample = factor_time_upsample
        self.freqs = freqs
        self.is_dedispersed = is_dedispersed
        self.is_folded = is_folded
        self.num_components = num_components
        self.scintillation = scintillation
        self.times = times
        self.verbose = verbose
        self.save = save_intermediate

        # derive some additional data needed for downstream fitting.
        self.num_freq = len(self.freqs)
        self.num_time = self.times.shape[-1]
        self.res_freq = np.fabs(self.freqs[1] - self.freqs[0])
        self.res_time = np.fabs(self.times[..., 1] - self.times[..., 0])

        # determine if the times array is frequency dependent
        if self.times.ndim == 1:
            self.has_freq_dependent_times = False

        elif self.times.ndim == 2:
            if self.times.shape[0] != self.num_freq:
                raise RuntimeError(
                    "times array has more than one dimension, "
                    "but first dimension does not match frequency axis."
                )

            self.has_freq_dependent_times = True

        else:
            raise RuntimeError(
                "Do not recognize shape of times array. "
                "Must be either (ntime,) or (nfreq, ntime)."
            )

        # define all *fittable* model parameters first.
        # NOTE: 'ref_freq' is not listed here as it's a parameter that is always held fixed.
        self.parameters = [
            "amplitude",
            "arrival_time",
            "burst_width",
            "dm",
            "dm_index",
            "scattering_timescale",
            "scattering_index",
            "spectral_index",
            "spectral_running",
        ]

        # now instantiate parameter attributes and set initially to NoneType.
        for current_parameter in self.parameters:
            setattr(self, current_parameter, None)

        # Initialize the slices here.
        self.actual_workers = min(num_threads, self.num_components)
        split_indices = np.array_split(
            np.arange(self.num_components), self.actual_workers
        )

        self.slices = [
            ModelSlice(
                self.freqs,
                self.times,
                int(arr[0]),
                int(arr[-1]) + 1,
                idx,
                self.dm_incoherent,
                self.factor_freq_upsample,
                self.factor_time_upsample,
                self.is_dedispersed,
                self.is_folded,
                self.scintillation,
                self.save,
            )
            for idx, arr in enumerate(split_indices)
            if len(arr > 0)
        ]

    def compute_model(self, data: float | None = None) -> float:
        if self.verbose:
            for current_component in range(self.num_components):
                current_amplitude = self.amplitude[current_component]
                current_arrival_time = self.arrival_time[current_component]
                current_dm = self.dm[0]
                current_sc_idx = self.scattering_index[0]
                current_sc_time = self.scattering_timescale[0]
                current_sp_idx = self.spectral_index[current_component]
                current_sp_run = self.spectral_running[current_component]
                current_width = self.burst_width[current_component]

                if self.scintillation:
                    print(
                        f"{current_dm:.5f} {current_arrival_time:.5f} ",
                        f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f}",
                    )
                else:
                    print(
                        f"{current_dm:.5f}  {current_amplitude:.5f}  {current_arrival_time:.5f}  ",
                        f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f} {current_sp_idx:.5f}  {current_sp_run:.5f}",
                    )

        def work(slice_object: ModelSlice):
            return slice_object.compute_model(data)

        with ThreadPoolExecutor(max_workers=self.actual_workers) as executor:
            results = list(executor.map(work, self.slices))

        full_spectrum = np.concatenate(results, axis=0)
        return np.sum(full_spectrum, axis=0)

    def get_parameters_dict(self) -> dict:
        """
        Returns model parameters as a dictionary, with keys set to the parameter names
        and values set to the Python list containing parameter values.

        Parameters
        ----------
        None : NoneType
            this method uses existing class attributes

        Returns
        -------
        parameter_dict : dict
            A dictionary containing parameter names as keys, and lists of per-component
            values as the dictionary values.
        """

        parameter_dict = {}
        for current_parameter in self.parameters:
            parameter_dict[current_parameter] = []

        # loop over all fittable parameters and grab their values.
        # for current_parameter in self.parameters:
        #     parameter_dict[current_parameter] = getattr(self, current_parameter)

        for slice in self.slices:
            for p, v in slice.parameters.items():
                parameter_dict[p].extend(v)

        # before exiting, grab the values of the reference frequency, which
        # isn't fittable and is therefore not in the 'parameters' list.
        parameter_dict["ref_freq"] = self.ref_freq

        return parameter_dict

    def update_parameters(
        self,
        model_parameters: dict,
        global_parameters: list | None = None,
    ) -> None:
        """
        Overloads parameter values stored in object with those supplied by the user.

        Parameters
        ----------
        model_parameters : dict
            a Python dictionary with parameter names listed as keys, parameter values
            supplied as lists tied to keys.

        global_parameters : list = ["dm", "scattering_timescale"]
            a Python list with the parameter names that are kept constant across components.

        Returns
        -------
        None : None
            this method overloads class attributes.
        """

        # first, overload attributes with values for supplied parameters.
        if global_parameters is None:
            global_parameters = ["dm", "scattering_timescale"]
        for current_parameter, current_value in model_parameters.items():
            setattr(self, current_parameter, current_value)

            # if number of components is 2 or greater, update num_components attribute.
            if len(current_value) > 1:
                num_components = len(current_value)
                self.num_components = num_components

                if current_parameter in global_parameters:
                    setattr(
                        self,
                        current_parameter,
                        [current_value[0]] * num_components,
                    )

        # Then let's update the slices:
        for slice in self.slices:
            start_idx = slice.start_idx
            end_idx = slice.end_idx

            new_params = {k: v[start_idx:end_idx] for k, v in model_parameters.items()}

            slice.update_parameters(new_params, global_parameters)
