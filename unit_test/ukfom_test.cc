#define BOOST_TEST_MODULE template_for_test_test
#include <boost/test/included/unit_test.hpp>

#include "boost_check_close_vector.hh"

#include "random_vector.hh"

#include <iostream>

// MTK's pose and orientation definition:
#include <mtk/types/pose.hpp>
#include <mtk/types/SOn.hpp>

// Main SLoM class:
#include <slom/Estimator.hpp>
// CallBack class for progress/debug output
#include <slom/CallBack.hpp>
// AutoConstruct macros to define measurements
#include <slom/BuildMeasurement.hpp>
// CholeskyCovariance to apply Covariance/Information-matrix
#include <slom/CholeskyCovariance.hpp>
// Simple timing class
#include <slom/TicToc.hpp>
//UKF class with Manifolds
#include <ukfom/ukf.hpp>
//UKF wrapper for the state vector
#include <ukfom/mtkwrap.hpp>

#include <vector>


#define MTK_DEFINE_FEATURE_TYPE(type)\
    typedef std::vector< type > MTK##type##Feature; \
    std::ostream& operator<<(std::ostream& __os, const MTK##type##Feature& __var) \
    { \
        __os << "["; \
        for (MTK##type##Feature::const_iterator ii = __var.begin(); ii != __var.end(); ++ii) \
        { \
            __os << " " << *ii; \
        } \
        __os << " ]"; \
        return __os; \
    } \


#define MTK_FEATURE_TYPE(type)\
    MTK##type##Feature \

// We can't use types having a comma inside AutoConstruct macros :(
typedef MTK::vect<3, double> vec3;
typedef MTK::SO3<double> SO3;
typedef std::vector<vec3> featureType;
MTK_DEFINE_FEATURE_TYPE(vec3)

MTK_BUILD_MANIFOLD ( PoseState ,
(( vec3 ,pos ))
(( SO3, orient ))
(( vec3, vel ))
(( vec3, gbias ))
(( vec3, abias ))
);

struct State
{
    typedef State self;


    MTK::SubManifold<PoseState, 0> posek;
    MTK::SubManifold<PoseState, PoseState::DOF + 0> posek_l;
    MTK::SubManifold<PoseState, PoseState::DOF + PoseState::DOF + 0> posek_i;
    MTK::SubManifold<MTK_FEATURE_TYPE(vec3), PoseState::DOF + PoseState::DOF + PoseState::DOF + 0> featuresk;
    MTK::SubManifold<MTK_FEATURE_TYPE(vec3), PoseState::DOF + PoseState::DOF + PoseState::DOF + 0> featuresk_l;

    enum
    {
        DOF = PoseState::DOF + PoseState::DOF + PoseState::DOF + 0
    };

    typedef vec3::scalar scalar;

    State ( const PoseState& posek = PoseState(),
            const PoseState& posek_l = PoseState(),
            const PoseState& posek_i = PoseState(),
            const MTK_FEATURE_TYPE(vec3)& featuresk = MTK_FEATURE_TYPE(vec3)(),
            const MTK_FEATURE_TYPE(vec3)& featuresk_l = MTK_FEATURE_TYPE(vec3)()
            )
        : posek(posek), posek_l(posek_l), posek_i(posek_i), featuresk(featuresk), featuresk_l(featuresk_l) {}

    int getDOF() const
    {
        return DOF;
    }

    void boxplus(const MTK::vectview<const scalar, DOF> & __vec, scalar __scale = 1 )
    {
        posek.boxplus(MTK::subvector(__vec, &self::posek), __scale);
        posek_l.boxplus(MTK::subvector(__vec, &self::posek_l), __scale);
        posek_i.boxplus(MTK::subvector(__vec, &self::posek_i), __scale);

    }

    void boxminus(MTK::vectview<scalar,DOF> __res, const State& __oth) const
    {
        posek.boxminus(MTK::subvector(__res, &self::posek), __oth.posek);
        posek_l.boxminus(MTK::subvector(__res, &self::posek_l), __oth.posek_l);
        posek_i.boxminus(MTK::subvector(__res, &self::posek_i), __oth.posek_i);
    }


    friend std::ostream& operator<<(std::ostream& __os, const State& __var)
    {
        __os << "\n" << __var.posek << "\n" << __var.posek_l << "\n" << __var.posek_i << "\n";
        __os << "[";
        for (MTK_FEATURE_TYPE(vec3)::const_iterator ii = __var.featuresk.begin(); ii != __var.featuresk.end(); ++ii)
        {
            __os << " " << *ii;
        }
        __os << " ]\n";
        __os << "[";
        for (MTK_FEATURE_TYPE(vec3)::const_iterator ii = __var.featuresk_l.begin(); ii != __var.featuresk_l.end(); ++ii)
        {
            __os << " " << *ii;
        }
        __os << " ]\n";

        return __os;
    }
    friend std::istream& operator>>(std::istream& __is, State& __var)
    {
        return __is >> __var.posek >> __var.posek_l >> __var.posek_i;
    }
};


// Alternatively use predefined trafo-Type (already implements transformations):
typedef MTK::trafo<MTK::SO2<double> > Pose;


// Define a measurement
SLOM_BUILD_MEASUREMENT(Odo, Pose::DOF, /* Name and dimension of measurement */
    ((Pose, t0))    /* ((type, name)) for each variable */ 
    ((Pose, t1))    /* type must be an MTK manifold */
    ,               /* Comma between variables and data */
    ((Pose, odo))   /* ((type, name)) for each data element */
    /* Type can be arbitrary, i.e. also pointers or references */
    ((SLOM::CholeskyCovariance<Pose::DOF>, cov)) 
);

typedef ukfom::mtkwrap<PoseState> UkfState;
typedef ukfom::mtkwrap<State> UkfTestingState;

//a is acceleration
//w is angular velocity (gyros)
PoseState process_model ( const PoseState &s, const Eigen::Matrix<double, 3, 1> &a , const MTK::vect <3,double > &w, double dt)
{
    PoseState s2 ;

    // apply rotation
    std::cout<<"w is: "<<w<<"\n";
    //MTK::vect <3, double> scaled_axis = w * dt ;
    Eigen::Vector3d scaled_axis = w * dt ;
    MTK::SO3 <double> rot = SO3::exp (scaled_axis);
    s2.orient = s.orient * rot;

    // accelerate with gravity
    //Eigen::Matrix<double, 3, 1> gravity;
    MTK::vect<3, double> gravity;
    gravity <<0,0,9.81;
    s2.vel = s.vel + (s.orient * a + gravity) * dt;

    // translate
    s2.pos = s.pos + s.vel * dt;

    std::cout<<"[PROCESS_MODEL] pos:"<<s2.pos<<" orient:"<<s2.orient<<" vel:"<<s2.vel<<"\n";

    return s2;
}


ukfom::ukf <UkfState>::cov process_noise_cov (double dt)
{
    ukfom::ukf<UkfState>::cov cov = ukfom::ukf<UkfState>::cov::Zero();
    setDiagonal (cov, &UkfState::pos, 0);
    setDiagonal (cov, &UkfState::orient,  0.0001 * dt);
    setDiagonal (cov, &UkfState::vel, 0.0002 * dt);
    setDiagonal (cov, &UkfState::gbias, 0.0002 * dt);
    setDiagonal (cov, &UkfState::abias, 0.0002 * dt);

    return cov ;
}

MTK::vect <3, double> gps_measurement_model ( const PoseState &s )
{
    return s.pos ;
}

Eigen::Matrix<double, 3, 3> gps_measurement_noise_cov ()
{
    return 0.00000001 * Eigen::Matrix<double, 3, 3>::Identity();
}


BOOST_AUTO_TEST_CASE( UKFOM )
{

    const UkfState ukf_state;
    const UkfTestingState myState;

    std::vector<int> hola(10,10);
    std::vector<int> bye (10,0);

    bye = hola;

    std::cout<<"ukf_state::DOF is "<<ukf_state.DOF<<"\n";
    std::cout<<"ukf_state.pos is "<<ukf_state.pos<<"\n";
    std::cout<<"ukf_state.orient is "<<ukf_state.orient<<"\n";
    std::cout<<"ukf_state.vel is "<<ukf_state.vel<<"\n";
    std::cout<<"ukf state is "<< ukf_state <<"\n";

    std::cout<<"myState::DOF is "<<myState.DOF<<"\n";
    std::cout<<"myState is "<< myState <<"\n";
    std::cout<<"myState.featuresk: "<<myState.featuresk <<"\n";

    const ukfom::ukf<UkfState>::cov init_cov = 0.001 * ukfom::ukf<UkfState>::cov::Identity();
    ukfom::ukf<UkfState> kf(ukf_state, init_cov);
    Eigen::Vector3d acc, gyro, gps;
    gyro<<0.1,0.0, 0.0;
    acc<<0.0, 0.0, 0.0;
    gps<<1.0, 0.0, 0.0;
    for (register int i=0; i<1; ++i)
    {
        std::cout<<"IN_LOOP["<<i<<"]\n";
        //boost::bind(process_model, _1 , acc, gyro, 0.01);
        //kf.predict();
        kf.predict(boost::bind(process_model, _1 , acc , gyro, 0.01), process_noise_cov(0.01));
        kf.update (gps, boost::bind(gps_measurement_model, _1), gps_measurement_noise_cov);
    }

}
